import json
import re
from typing import Dict, List, Any, Optional
from app.core.logging_config import logger


SYSTEM_TABLE_PATTERNS = [
    "migrations",
    "failed_jobs",
    "jobs",
    "job_batches",
    "sessions",
    "cache",
    "cache_locks",
    "password_reset_tokens",
    "password_resets",
    "personal_access_tokens",
    "telescope_entries",
    "telescope_entries_tags",
    "telescope_monitoring",
    "horizon_jobs",
    "pulse_entries",
    "pulse_values",
    "pulse_aggregates",
    "oauth_access_tokens",
    "oauth_auth_codes",
    "oauth_clients",
    "oauth_personal_access_clients",
    "oauth_refresh_tokens",
    "notifications",
    "queue_jobs",
    "delayed_jobs",
    "ar_internal_metadata",
    "schema_migrations",
    "django_migrations",
    "django_content_type",
    "django_session",
    "auth_permission",
    "auth_group",
]

SENSITIVE_COL_PATTERNS = [
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "private_key",
    "access_token",
    "refresh_token",
    "salt",
    "hash",
    "encrypted",
    "ssn",
    "credit_card",
    "cvv",
    "pin",
]


def extract_json(text: str) -> Any:
    """
    Robustly extract the first valid JSON object OR array from LLM output.
    Uses bracket-counting so it stops at the correct closing brace
    — not greedy regex which breaks on extra text after the JSON.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx = text.find(start_char)
        if idx == -1:
            continue
        depth, in_string, escape_next = 0, False, False
        for i, ch in enumerate(text[idx:], start=idx):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx : i + 1])
                    except json.JSONDecodeError:
                        break
    logger.error(f"Could not extract JSON from LLM response: {text[:300]}")
    return None


# ── Groq-first LLM caller ────────────────────────────────────────────────────
# We use Groq by default for schema analysis — Gemini's RPM limit is too low
# and schema analysis fires one call per selected table + narrative + actions.
# Groq (llama-3.3-70b) handles JSON reliably and has no RPM problem for this.

from app.services.llm import LLMService


async def _call_llm_raw_impl(self, prompt: str, prefer_groq: bool = True) -> str:
    """Call LLM and return raw text. Defaults to Groq to avoid Gemini RPM limits."""
    messages = [
        {
            "role": "system",
            "content": "Return only valid JSON. No markdown fences, no explanation, just the JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        if prefer_groq:
            # Groq-first: only fall back to Gemini if Groq fails
            try:
                return await self._generate_with_groq(messages)
            except Exception as groq_err:
                logger.warning(f"Groq failed, falling back to Gemini: {groq_err}")
                reply = await self._generate_with_gemini(messages)
                self.rate_limiter.add_request("gemini")
                return reply
        else:
            # Original logic (Gemini-first) — kept for non-schema calls
            use_groq = self._should_use_groq()
            if use_groq:
                return await self._generate_with_groq(messages)
            try:
                reply = await self._generate_with_gemini(messages)
                self.rate_limiter.add_request("gemini")
                return reply
            except Exception:
                return await self._generate_with_groq(messages)
    except Exception as e:
        logger.error(f"LLM raw call failed: {e}")
        return "{}"


if not hasattr(LLMService, "_call_llm_raw"):
    LLMService._call_llm_raw = _call_llm_raw_impl


# ─────────────────────────────────────────────────────────────────────────────
# ProfileBuilder
# ─────────────────────────────────────────────────────────────────────────────


class ProfileBuilder:
    def __init__(self, llm_service):
        self.llm = llm_service

    async def build(
        self,
        connection_id: str,
        business_id: str,
        chatbot_id: str,
        database_type: str,
        all_tables: List[Dict[str, Any]],
        selected_tables: List[str],
        table_profiles: Dict[str, Any],
    ) -> Dict[str, Any]:

        logger.info(
            f"ProfileBuilder.build() — {len(selected_tables)} tables, type={database_type}"
        )

        # 1. Classify all tables
        table_classifications = self._classify_all_tables(all_tables, selected_tables)

        # 2. Detect global context clues before per-table analysis
        global_context = self._build_global_context(selected_tables, table_profiles)

        # 3. Analyze each selected table
        table_entity_types = {}
        # customer_identifiers: Dict[table -> List[str]]  (multiple valid identifiers per table)
        customer_identifiers = {}
        relationships = []

        for table in selected_tables:
            profile = table_profiles.get(table, {})
            analysis = await self._analyze_table(
                table, profile, selected_tables, database_type, global_context
            )

            table_entity_types[table] = analysis.get("entity_type", "unknown")
            # Always store as list — deduplicated, non-null
            raw_ids = analysis.get("customer_identifiers") or []
            if not isinstance(raw_ids, list):
                raw_ids = [raw_ids]
            # Also accept legacy single "customer_identifier" key
            single = analysis.get("customer_identifier")
            if single and single not in raw_ids:
                raw_ids.append(single)
            customer_identifiers[table] = [
                v for v in dict.fromkeys(raw_ids) if v
            ]  # dedup, remove None

            for rel in analysis.get("relationships", []):
                if rel not in relationships:
                    relationships.append(rel)

            db_fks = profile.get("foreign_keys", [])
            ai_fks = analysis.get("foreign_keys", [])
            profile["foreign_keys"] = self._merge_foreign_keys(
                db_fks, ai_fks, selected_tables
            )

        # 4. Infer business domain via LLM — runs after entity types are known
        #    Updates global_context in-place (adds business_domain + summary)
        await self._infer_business_domain(
            global_context,
            table_entity_types,
            selected_tables,
            table_profiles,
            database_type,
        )

        # 5. Business narrative
        narrative = await self._generate_narrative(
            database_type,
            selected_tables,
            table_profiles,
            table_entity_types,
            relationships,
            global_context,
        )

        # 6. Discovered actions
        discovered_actions = await self._discover_actions(
            database_type,
            selected_tables,
            table_profiles,
            table_entity_types,
            narrative,
        )

        # 7. Schema summary
        schema_summary = {}
        for table in selected_tables:
            p = table_profiles.get(table, {})
            schema_summary[table] = {
                "columns": p.get("columns", []),
                "foreign_keys": p.get("foreign_keys", []),
                "enum_hints": p.get("enum_hints", {}),
                "entity_type": table_entity_types.get(table, "unknown"),
                "customer_identifiers": customer_identifiers.get(table, []),
                "sample_values": self._sanitize_samples(p.get("sample_rows", [])[:3]),
            }

        return {
            "connection_id": connection_id,
            "business_id": business_id,
            "chatbot_id": chatbot_id,
            "database_type": database_type,
            "selected_tables": selected_tables,
            "table_classifications": table_classifications,
            "table_entity_types": table_entity_types,
            "customer_identifiers": customer_identifiers,
            "relationships": relationships,
            "schema_summary": schema_summary,
            "business_type": narrative.get("business_type", "Unknown"),
            "business_narrative": narrative.get("narrative", ""),
            "discovered_actions": discovered_actions,
            "global_context": global_context,
            "ai_generated": True,
        }

    # ─── Global context detection ─────────────────────────────────────────────
    # This runs BEFORE per-table analysis so every LLM prompt can reason
    # from a shared understanding of the app type.

    def _build_global_context(
        self, selected_tables: List[str], table_profiles: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 1 — structural signals only, no LLM, no keyword guessing.
        Computes facts that are deterministically readable from the schema data.

        New shape (replaces the old brittle heuristic fields):
          ownership_pattern : "single_owner" | "multi_tenant" | "shared" | "unknown"
          user_table        : fully-qualified table name or None
          user_row_estimate : int | None
          has_explicit_fk   : bool  — any content table has user_id / owner_id column
          estimated_row_counts : {table: int}
          business_domain   : None  (filled later by _infer_business_domain)
          summary           : ""    (filled later by _infer_business_domain)
        """
        context: Dict[str, Any] = {
            "ownership_pattern": "unknown",
            "user_table": None,
            "user_row_estimate": None,
            "has_explicit_fk": False,
            "estimated_row_counts": {},
            "business_domain": None,  # filled by _infer_business_domain
            "summary": "",  # filled by _infer_business_domain
        }

        for table in selected_tables:
            profile = table_profiles.get(table, {})
            sample_rows = profile.get("sample_rows", [])
            bare = table.lower().split(".")[-1]
            estimated = len(sample_rows)
            context["estimated_row_counts"][table] = estimated

            # Identity/user table detection — structural, not keyword-guessing
            if any(
                k in bare
                for k in ("user", "customer", "client", "account", "member", "tenant")
            ):
                context["user_table"] = table
                context["user_row_estimate"] = estimated

            # Explicit ownership FK detection
            cols = [c["name"].lower() for c in profile.get("columns", [])]
            if any(
                c in cols
                for c in (
                    "user_id",
                    "owner_id",
                    "customer_id",
                    "tenant_id",
                    "account_id",
                )
            ):
                context["has_explicit_fk"] = True

        # Ownership pattern — deterministic from structural signals
        has_user = context["user_table"] is not None
        row_est = context["user_row_estimate"]

        if (
            has_user
            and row_est is not None
            and row_est <= 2
            and not context["has_explicit_fk"]
        ):
            context["ownership_pattern"] = "single_owner"
        elif has_user and context["has_explicit_fk"]:
            context["ownership_pattern"] = "multi_tenant"
        elif has_user:
            context["ownership_pattern"] = "shared"
        else:
            context["ownership_pattern"] = "unknown"

        logger.info(
            f"Global context (structural): ownership={context['ownership_pattern']}, "
            f"user_table={context['user_table']}, user_rows={context['user_row_estimate']}, "
            f"explicit_fk={context['has_explicit_fk']}"
        )
        return context

    async def _infer_business_domain(
        self,
        global_context: Dict[str, Any],
        table_entity_types: Dict[str, str],
        selected_tables: List[str],
        table_profiles: Dict[str, Any],
        database_type: str,
    ) -> None:
        """
        Phase 2 — one LLM call to infer business_domain and write the summary.
        Runs AFTER per-table analysis so table_entity_types are available.
        Updates global_context in-place.

        business_domain examples (not exhaustive — LLM decides):
          personal_portfolio, e_commerce, saas_platform, blog_cms, healthcare,
          education, booking_reservations, inventory_management, social_network,
          analytics_dashboard, crm, event_management, real_estate, finance,
          restaurant, logistics, hr_payroll, project_management, ...
        """
        # Build a compact description for the LLM
        entity_summary = ", ".join(
            f"{t.split('.')[-1]}({e})" for t, e in table_entity_types.items()
        )

        # Include a few sample values from each selected table (non-sensitive only)
        sample_glimpse = []
        for table in selected_tables[:4]:  # cap at 4 tables
            p = table_profiles.get(table, {})
            rows = p.get("sample_rows", [])[:1]
            if rows:
                clean = {
                    k: v
                    for k, v in rows[0].items()
                    if not self._is_sensitive_col(k) and v is not None
                }
                sample_glimpse.append(
                    f"{table.split('.')[-1]}: {list(clean.keys())[:6]}"
                )

        ownership_note = {
            "single_owner": "single-owner / personal app (1 user, owns all content)",
            "multi_tenant": "multi-tenant (many users, each owns their records)",
            "shared": "shared content (users exist, unclear ownership pattern)",
            "unknown": "no clear user/owner table detected",
        }.get(global_context["ownership_pattern"], "unknown")

        prompt = f"""Identify the business domain of this database. Return ONLY a JSON object.

Database type: {database_type}
Tables and entity types: {entity_summary}
Ownership pattern: {ownership_note}
Sample column names:
{chr(10).join(sample_glimpse)}

Return exactly this JSON (no markdown, no extra keys):
{{
  "business_domain": "personal_portfolio",
  "confidence": "high",
  "summary": "One sentence describing what this app does and its ownership pattern."
}}

Rules:
- business_domain: a snake_case identifier that accurately describes the business
  e.g. personal_portfolio, e_commerce, saas_platform, blog_cms, healthcare_clinic,
       education_lms, booking_system, inventory_management, social_network, crm,
       restaurant_pos, logistics, hr_payroll, project_management, real_estate,
       finance_tracker, event_management — or invent a new accurate one
- confidence: "high" | "medium" | "low"
- summary: inject ownership pattern info, e.g. "single-owner portfolio" vs "multi-tenant SaaS"
"""

        try:
            raw = await self.llm._call_llm_raw(prompt)
            result = extract_json(raw)
            if isinstance(result, dict) and "business_domain" in result:
                global_context["business_domain"] = result["business_domain"]
                raw_summary = result.get("summary", "")

                # Ownership pattern prefix — prepended so LLM prompts get
                # the structural fact even if the AI summary omits it
                ownership_prefix = {
                    "single_owner": (
                        f"CONTEXT: Single-owner app — the '{global_context['user_table']}' table "
                        f"has only {global_context['user_row_estimate']} row(s), so one user "
                        "implicitly owns all content. Infer this ownership in relationships[] "
                        "with note='inferred-single-owner'."
                    ),
                    "multi_tenant": (
                        "CONTEXT: Multi-tenant system — user_id foreign keys define record "
                        "ownership. Always filter by the authenticated user's ID."
                    ),
                    "shared": (
                        "CONTEXT: Shared-content system — users exist but content may not "
                        "be strictly owned per-user."
                    ),
                    "unknown": "",
                }.get(global_context["ownership_pattern"], "")

                global_context["summary"] = (
                    f"{ownership_prefix} Business domain: {result['business_domain']}. {raw_summary}"
                ).strip()

                logger.info(
                    f"Business domain inferred: {result['business_domain']} "
                    f"(confidence: {result.get('confidence', '?')})"
                )
                return

        except Exception as e:
            logger.warning(f"Business domain inference failed: {e}")

        # Fallback — at least write the structural ownership context
        global_context["business_domain"] = "unknown"
        ownership_fallback = {
            "single_owner": (
                f"CONTEXT: Single-owner app — '{global_context['user_table']}' has "
                f"{global_context['user_row_estimate']} row(s). One user owns all content implicitly. "
                "Set relationships[] with note='inferred-single-owner'."
            ),
            "multi_tenant": (
                "CONTEXT: Multi-tenant — filter all queries by authenticated user_id."
            ),
            "shared": "CONTEXT: Shared content system.",
            "unknown": "",
        }.get(global_context["ownership_pattern"], "")
        global_context["summary"] = ownership_fallback

    # ─── Table classification ─────────────────────────────────────────────────

    def _classify_all_tables(
        self, all_tables: List[Dict], selected_tables: List[str]
    ) -> Dict[str, str]:
        classifications = {}
        selected_lower = {t.lower() for t in selected_tables}
        if all_tables:
            for t in all_tables:
                name = t["name"]
                lower = name.lower()
                if lower in selected_lower:
                    classifications[name] = "selected"
                elif t.get("likely_system") or self._is_system_table(lower):
                    classifications[name] = "system"
                else:
                    classifications[name] = "business"
        else:
            for table in selected_tables:
                bare = table.lower().split(".")[-1]
                classifications[table] = (
                    "system" if self._is_system_table(bare) else "selected"
                )
        return classifications

    def _is_system_table(self, table_lower: str) -> bool:
        bare = table_lower.split(".")[-1]
        return any(bare == p or bare.startswith(p) for p in SYSTEM_TABLE_PATTERNS)

    # ─── Per-table analysis ───────────────────────────────────────────────────

    async def _analyze_table(
        self,
        table: str,
        profile: Dict[str, Any],
        all_selected: List[str],
        database_type: str,
        global_context: Dict[str, Any],
    ) -> Dict[str, Any]:

        columns = profile.get("columns", [])
        sample_rows = profile.get("sample_rows", [])
        db_fks = profile.get("foreign_keys", [])
        enum_hints = profile.get("enum_hints", {})

        col_descriptions = []
        for c in columns[:25]:
            desc = f"{c['name']} ({c['type']})"
            if c.get("default"):
                desc += f" [default: {c['default']}]"
            if c["name"] in enum_hints:
                vals = enum_hints[c["name"]][:5]
                desc += f" [values: {', '.join(str(v) for v in vals)}]"
            col_descriptions.append(desc)

        sample_text = ""
        if sample_rows:
            sample_text = "\nSAMPLE DATA (sensitive fields removed):\n"
            for i, row in enumerate(sample_rows[:3]):
                clean = {k: v for k, v in row.items() if not self._is_sensitive_col(k)}
                sample_text += (
                    f"  Row {i + 1}: {json.dumps(clean, default=str)[:400]}\n"
                )

        other_tables = [t for t in all_selected if t != table]

        # Build ownership context from structural signal — available immediately,
        # unlike global_context["summary"] which is filled later by _infer_business_domain
        ownership = global_context.get("ownership_pattern", "unknown")
        user_tbl = global_context.get("user_table")
        user_rows = global_context.get("user_row_estimate")

        if ownership == "single_owner" and user_tbl:
            ownership_note = (
                f"IMPORTANT: This is a SINGLE-OWNER app — '{user_tbl}' has only {user_rows} row(s). "
                f"One user implicitly owns ALL content tables, even without an explicit FK. "
                f"Add an inferred relationship to '{user_tbl}' in relationships[] with "
                f"note='inferred-single-owner' for every content table."
            )
        elif ownership == "multi_tenant":
            ownership_note = (
                "CONTEXT: This is a MULTI-TENANT system. "
                "user_id / owner_id foreign keys define per-user record ownership."
            )
        else:
            ownership_note = ""

        # Bare table name hint helps LLM avoid confusing e.g. "project_images" → "projects"
        bare_name = table.split(".")[-1]

        prompt = f"""Analyze this database table. Return ONLY a valid JSON object — no markdown, no explanation.

{ownership_note}

Table: {table}  (bare name: {bare_name})
Database type: {database_type}
Other selected tables: {", ".join(other_tables) if other_tables else "none"}

Columns:
{chr(10).join(col_descriptions)}

Known FK constraints from DB: {json.dumps(db_fks)}
{sample_text}
Return exactly this JSON shape:
{{
  "entity_type": "project_images",
  "primary_key": "id",
  "customer_identifiers": [],
  "foreign_keys": [],
  "relationships": []
}}

Rules:
- entity_type: what this table ACTUALLY stores — use the bare table name as your primary hint.
  e.g. bare name 'project_images' → entity_type 'project_images' (NOT 'projects')
  e.g. bare name 'order_items' → entity_type 'order_items' (NOT 'orders')
  Keep it specific — don't generalise to the parent entity.
- customer_identifiers: LIST of columns identifying the owner/user. Return [] if none.
- foreign_keys and relationships MUST ONLY reference tables from: {json.dumps(other_tables)}
- If ownership_note above says single-owner: add inferred relationship to the owner table
  in relationships[] with note='inferred-single-owner' for EVERY content table here.
"""

        try:
            raw = await self.llm._call_llm_raw(prompt)
            analysis = extract_json(raw)
            if not isinstance(analysis, dict):
                raise ValueError("Not a dict")
            analysis = self._normalize_customer_identifiers(analysis)
            return self._validate_table_analysis(analysis, all_selected)
        except Exception as e:
            logger.warning(f"Table analysis failed for {table}: {e}")
            return self._fallback_table_analysis(table, columns, db_fks, all_selected)

    def _normalize_customer_identifiers(self, analysis: Dict) -> Dict:
        """Ensure customer_identifiers is always a list, handle legacy single-value key."""
        raw = analysis.get("customer_identifiers") or []
        if not isinstance(raw, list):
            raw = [raw]
        single = analysis.get("customer_identifier")
        if single and single not in raw:
            raw.append(single)
        analysis["customer_identifiers"] = [v for v in dict.fromkeys(raw) if v]
        # Remove legacy key
        analysis.pop("customer_identifier", None)
        return analysis

    def _validate_table_analysis(self, analysis: Dict, valid_tables: List[str]) -> Dict:
        def table_is_valid(ref: str) -> bool:
            ref_lower = ref.lower()
            return ref_lower in {t.lower() for t in valid_tables} or any(
                ref_lower == t.lower().split(".")[-1] for t in valid_tables
            )

        analysis["foreign_keys"] = [
            fk
            for fk in analysis.get("foreign_keys", [])
            if table_is_valid(fk.get("references_table", ""))
        ]
        analysis["relationships"] = [
            r
            for r in analysis.get("relationships", [])
            if table_is_valid(r.get("table", ""))
        ]
        return analysis

    def _fallback_table_analysis(
        self,
        table: str,
        columns: List[Dict],
        db_fks: List[Dict],
        all_selected: List[str],
    ) -> Dict:
        identifiers = []
        primary_key = "id"
        for col in columns:
            name_lower = col["name"].lower()
            if name_lower in ["id", "_id"]:
                primary_key = col["name"]
            if name_lower in ["user_id", "customer_id", "client_id", "email"]:
                identifiers.append(col["name"])
        return {
            "entity_type": self._infer_entity_type(table),
            "primary_key": primary_key,
            "customer_identifiers": identifiers,
            "foreign_keys": db_fks,
            "relationships": [],
        }

    def _infer_entity_type(self, table: str) -> str:
        lower = table.lower().split(".")[-1]
        for keyword, entity in {
            "user": "users",
            "customer": "users",
            "client": "users",
            "order": "orders",
            "product": "products",
            "item": "products",
            "payment": "payments",
            "transaction": "payments",
            "invoice": "payments",
            "project": "projects",
            "image": "images",
            "media": "media",
            "blog": "blog_posts",
            "post": "blog_posts",
            "article": "blog_posts",
            "subscription": "subscriptions",
            "report": "reports",
        }.items():
            if keyword in lower:
                return entity
        return lower

    # ─── Business narrative ───────────────────────────────────────────────────

    async def _generate_narrative(
        self,
        database_type: str,
        selected_tables: List[str],
        table_profiles: Dict,
        table_entity_types: Dict,
        relationships: List[Dict],
        global_context: Dict,
    ) -> Dict[str, Any]:

        tables_summary = []
        for table in selected_tables:
            p = table_profiles.get(table, {})
            entity = table_entity_types.get(table, "unknown")
            col_names = [c["name"] for c in p.get("columns", [])[:8]]
            sample = p.get("sample_rows", [])
            tables_summary.append(
                f"- {table} ({entity}): {', '.join(col_names)} ({len(sample)} sample rows)"
            )

        context_note = global_context.get("summary", "")

        prompt = f"""Analyze this database and return ONLY a JSON object.

{context_note}

Database type: {database_type}
Tables:
{chr(10).join(tables_summary)}

Return exactly:
{{
  "business_type": "Personal Portfolio Website",
  "narrative": "2-3 sentences describing what this app/system does, naming actual tables. Be specific about whether it's single-owner or multi-user based on the context above."
}}"""

        try:
            raw = await self.llm._call_llm_raw(prompt)
            result = extract_json(raw)
            if isinstance(result, dict) and "narrative" in result:
                return result
            raise ValueError("Missing narrative")
        except Exception as e:
            logger.warning(f"Narrative generation failed: {e}")
            return {
                "business_type": "Unknown",
                "narrative": f"A {database_type} database with tables: {', '.join(selected_tables)}.",
            }

    # ─── Discovered actions ───────────────────────────────────────────────────

    async def _discover_actions(
        self,
        database_type: str,
        selected_tables: List[str],
        table_profiles: Dict,
        table_entity_types: Dict,
        narrative: Dict,
    ) -> List[Dict]:

        prompt = f"""Based on this database, list realistic questions a chatbot visitor could ask.
Return ONLY a JSON array — no markdown, no extra text.

Business: {narrative.get("business_type", "Unknown")}
Tables and types: {json.dumps({t: table_entity_types.get(t) for t in selected_tables})}

[
  {{
    "key": "view_projects",
    "label": "View Projects",
    "description": "List available projects and details",
    "permission": "public",
    "tables_needed": {json.dumps(selected_tables[:1])},
    "example_question": "What projects have you built?"
  }}
]

Rules:
- tables_needed values must EXACTLY match entries from: {json.dumps(selected_tables)}
- Generate 3-6 realistic actions based on what the data actually supports
- permission: "public" or "authenticated"
"""

        try:
            raw = await self.llm._call_llm_raw(prompt)
            result = extract_json(raw)
            if isinstance(result, list):
                return self._validate_actions(result, selected_tables)
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        return self._validate_actions(v, selected_tables)
            return []
        except Exception as e:
            logger.warning(f"Action discovery failed: {e}")
            return []

    def _validate_actions(
        self, actions: List[Dict], valid_tables: List[str]
    ) -> List[Dict]:
        valid = []
        for a in actions:
            tables_needed = a.get("tables_needed", [])
            ok = all(
                any(needed == t or needed in t or t in needed for t in valid_tables)
                for needed in tables_needed
            )
            if ok or not tables_needed:
                valid.append(a)
        return valid

    # ─── FK merging ───────────────────────────────────────────────────────────

    def _merge_foreign_keys(
        self, db_fks: List[Dict], ai_fks: List[Dict], valid_tables: List[str]
    ) -> List[Dict]:
        merged = list(db_fks)
        existing_cols = {fk["column"] for fk in db_fks}
        for fk in ai_fks:
            col = fk.get("column")
            ref = fk.get("references_table", "")
            ref_valid = any(
                ref.lower() == t.lower() or ref.lower() == t.lower().split(".")[-1]
                for t in valid_tables
            )
            if col not in existing_cols and ref_valid:
                merged.append(fk)
        return merged

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _sanitize_samples(self, rows: List[Dict]) -> List[Dict]:
        result = []
        for row in rows:
            clean = {}
            for k, v in row.items():
                if self._is_sensitive_col(k):
                    continue
                if isinstance(v, str) and len(v) > 100:
                    v = v[:100] + "..."
                clean[k] = v
            result.append(clean)
        return result

    def _is_sensitive_col(self, col_name: str) -> bool:
        lower = col_name.lower()
        return any(p in lower for p in SENSITIVE_COL_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy classes
# ─────────────────────────────────────────────────────────────────────────────


class SchemaAnalyzer:
    def __init__(self, llm_service):
        self.llm = llm_service

    async def analyze_table_structure(
        self, table: str, columns: List[str], all_tables: List[str]
    ) -> Dict[str, Any]:
        prompt = f"""Analyze this table. Return ONLY valid JSON.
Table: {table}
Columns: {", ".join(columns)}
Other tables: {", ".join(all_tables)}

{{"primary_key": "id", "foreign_keys": [], "customer_identifiers": [], "entity_type": "unknown", "relationships": []}}"""
        try:
            raw = await self.llm._call_llm_raw(prompt)
            response = extract_json(raw)
            if isinstance(response, dict):
                response = self._normalize(response)
                return self._validate(response, all_tables)
            return self._fallback(table, columns, all_tables)
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            return self._fallback(table, columns, all_tables)

    def _normalize(self, analysis: Dict) -> Dict:
        """Normalize customer_identifier(s) to list."""
        raw = analysis.get("customer_identifiers") or []
        if not isinstance(raw, list):
            raw = [raw]
        single = analysis.get("customer_identifier")
        if single and single not in raw:
            raw.append(single)
        analysis["customer_identifiers"] = [v for v in dict.fromkeys(raw) if v]
        analysis.pop("customer_identifier", None)
        return analysis

    def _validate(self, analysis: Dict, valid_tables: List[str]) -> Dict:
        valid_lower = [t.lower() for t in valid_tables]
        analysis["foreign_keys"] = [
            fk
            for fk in analysis.get("foreign_keys", [])
            if fk.get("references_table", "").lower() in valid_lower
        ]
        analysis["relationships"] = [
            r
            for r in analysis.get("relationships", [])
            if r.get("table", "").lower() in valid_lower
        ]
        return analysis

    def _fallback(self, table: str, columns: List[str], all_tables: List[str]) -> Dict:
        identifiers = []
        all_lower = [t.lower() for t in all_tables]
        fks = []
        pk = "id"
        for col in columns:
            lower = col.lower()
            if lower in ["id", "_id"]:
                pk = col
            if lower.endswith("_id") and lower != "id":
                ref = lower.replace("_id", "")
                for c in [ref, ref + "s"]:
                    if c in all_lower:
                        fks.append(
                            {
                                "column": col,
                                "references_table": all_tables[all_lower.index(c)],
                            }
                        )
                        break
            if lower in ["user_id", "customer_id", "email"]:
                identifiers.append(col)
        return {
            "primary_key": pk,
            "foreign_keys": fks,
            "customer_identifiers": identifiers,
            "entity_type": "unknown",
            "relationships": [],
        }


class BusinessOverviewGenerator:
    def __init__(self, llm_service):
        self.llm = llm_service

    async def generate_overview(
        self,
        schema_analysis: Dict,
        database_type: str,
        selected_tables: Optional[List[str]] = None,
    ) -> Dict:
        selected = selected_tables or list(schema_analysis.get("tables", {}).keys())
        tables_summary = [
            f"- {t} ({info.get('entity_type', '?')}): {', '.join(c['name'] for c in info.get('columns', [])[:8])}"
            for t, info in schema_analysis.get("tables", {}).items()
        ]
        prompt = f"""Analyze this business database. Return ONLY valid JSON.
Database Type: {database_type}
Tables:
{chr(10).join(tables_summary)}

{{"business_type": "...", "overview": "...", "discovered_actions": []}}"""
        try:
            raw = await self.llm._call_llm_raw(prompt)
            response = extract_json(raw)
            if isinstance(response, dict):
                response["analyzed_tables"] = selected
                return response
            raise ValueError("invalid")
        except Exception:
            return {
                "business_type": "Unknown",
                "overview": f"Tables: {', '.join(selected)}",
                "discovered_actions": [],
                "analyzed_tables": selected,
            }
