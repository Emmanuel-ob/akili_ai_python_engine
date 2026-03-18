"""
InsightEngine — Point 5A

Uses column fingerprinting to identify table roles regardless of naming convention.
Works with WooCommerce, Shopify, custom schemas, etc.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from app.services.llm import LLMService
from app.services.sql_executor import DirectSQLExecutor
from app.core.logging_config import logger


# ─── Column fingerprints ──────────────────────────────────────────────────────
# (role, required_cols_any_of, bonus_cols)
_COLUMN_FINGERPRINTS = [
    (
        "orders",
        [
            "total_amount",
            "order_total",
            "grand_total",
            "amount",
            "price",
            "subtotal",
            "total_price",
            "sale_price",
            "order_value",
        ],
        ["status", "user_id", "customer_id", "created_at", "paid_at"],
    ),
    (
        "customers",
        [
            "email",
            "phone",
            "mobile",
            "customer_name",
            "first_name",
            "last_name",
            "full_name",
            "username",
        ],
        ["created_at", "user_id", "address", "city"],
    ),
    (
        "products",
        [
            "product_name",
            "item_name",
            "sku",
            "barcode",
            "stock_quantity",
            "stock",
            "inventory",
            "unit_price",
            "selling_price",
        ],
        ["category", "description", "image", "created_at"],
    ),
    (
        "order_items",
        [
            "product_id",
            "item_id",
            "quantity",
            "qty",
            "unit_price",
            "line_total",
            "item_price",
            "order_id",
        ],
        ["discount", "tax", "sku"],
    ),
    (
        "transactions",
        [
            "transaction_id",
            "reference",
            "ref_no",
            "payment_method",
            "gateway",
            "payment_status",
            "paid_amount",
            "charged_amount",
        ],
        ["created_at", "user_id", "status"],
    ),
]

_NAME_KEYWORDS = {
    "orders": [
        "order",
        "sale",
        "transaction",
        "purchase",
        "booking",
        "invoice",
        "bill",
        "receipt",
        "trx",
        "txn",
    ],
    "customers": [
        "customer",
        "client",
        "user",
        "member",
        "account",
        "subscriber",
        "buyer",
        "shopper",
    ],
    "products": [
        "product",
        "item",
        "goods",
        "stock",
        "catalog",
        "inventory",
        "article",
        "listing",
        "sku",
    ],
    "order_items": [
        "order_item",
        "line_item",
        "cart_item",
        "basket_item",
        "sale_item",
        "purchase_item",
    ],
    "transactions": [
        "transaction",
        "payment",
        "transfer",
        "charge",
        "remittance",
        "settlement",
    ],
}


def _get_col_names(table: str, schema: dict) -> List[str]:
    cols = schema.get(table, {}).get("columns", [])
    names = []
    for c in cols:
        if isinstance(c, dict):
            names.append((c.get("name") or c.get("column_name") or "").lower())
        elif isinstance(c, str):
            names.append(c.lower())
    return [n for n in names if n]


def _score(table: str, schema: dict, role: str) -> int:
    score = 0
    tname = table.lower()
    for kw in _NAME_KEYWORDS.get(role, []):
        if kw in tname:
            score += 10
            break
    cols = _get_col_names(table, schema)
    for fp_role, req, bonus in _COLUMN_FINGERPRINTS:
        if fp_role != role:
            continue
        score += sum(5 for r in req if r in cols)
        score += sum(2 for b in bonus if b in cols)
        break
    return score


def _resolve(role: str, schema: dict) -> Optional[str]:
    best, best_score = None, 0
    for t in schema.keys():
        s = _score(t, schema, role)
        if s > best_score:
            best, best_score = t, s
    if best_score >= 5:
        logger.info(f"InsightEngine: role={role} → {best} (score={best_score})")
        return best
    logger.info(f"InsightEngine: role={role} → no match")
    return None


def _amount_col(table: str, schema: dict) -> str:
    cols = _get_col_names(table, schema)
    for c in [
        "total_amount",
        "order_total",
        "grand_total",
        "amount",
        "total_price",
        "sale_price",
        "price",
        "subtotal",
        "paid_amount",
        "order_value",
        "value",
        "cost",
    ]:
        if c in cols:
            return c
    for c in cols:
        if any(kw in c for kw in ["amount", "price", "total", "value", "cost", "sum"]):
            return c
    return "1"


def _status_col(table: str, schema: dict) -> Optional[str]:
    cols = _get_col_names(table, schema)
    for c in ["status", "order_status", "state", "payment_status", "stage"]:
        if c in cols:
            return c
    return None


def _created_col(table: str, schema: dict) -> Optional[str]:
    cols = _get_col_names(table, schema)
    for c in [
        "created_at",
        "created",
        "date_created",
        "order_date",
        "transaction_date",
        "date",
        "timestamp",
        "inserted_at",
    ]:
        if c in cols:
            return c
    return None


class InsightEngine:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def generate_insights(
        self, database_config, database_profile, business_overview="", max_insights=6
    ):
        db_type = database_config.get("type", "postgresql")
        schema = database_profile.get("schema_summary") or {}
        tables = list(schema.keys())

        logger.info(f"InsightEngine: tables available = {tables}")

        if not tables:
            logger.warning("InsightEngine: no tables in schema_summary")
            return []

        roles = {
            r: _resolve(r, schema)
            for r in ["orders", "customers", "products", "order_items", "transactions"]
        }
        logger.info(f"InsightEngine: role map = {roles}")

        generators = [
            self._revenue,
            self._pending,
            self._new_customers,
            self._total_records,
            self._recent_activity,
            self._status_breakdown,
        ]

        insights = []
        for gen in generators:
            if len(insights) >= max_insights:
                break
            try:
                result = await gen(
                    db_type, schema, roles, database_config, business_overview
                )
                if result:
                    insights.append(result)
            except Exception as e:
                logger.error(
                    f"InsightEngine: {gen.__name__} failed: {e}", exc_info=True
                )

        logger.info(f"InsightEngine: generated {len(insights)} insights")
        return insights

    async def _revenue(self, db_type, schema, roles, db_cfg, biz_ov):
        table = roles.get("orders") or roles.get("transactions")
        if not table:
            return None
        acol = _amount_col(table, schema)
        ccol = _created_col(table, schema)
        if ccol:
            interval = (
                "NOW() - INTERVAL '30 days'"
                if db_type == "postgresql"
                else "DATE_SUB(NOW(), INTERVAL 30 DAY)"
            )
            sql = f"SELECT COALESCE(SUM({acol}), 0) AS value FROM {table} WHERE {ccol} >= {interval}"
        else:
            sql = f"SELECT COALESCE(SUM({acol}), 0) AS value FROM {table}"
        return await self._run(
            "revenue_30d",
            "Revenue (30 days)",
            "💰",
            "revenue",
            sql,
            db_cfg,
            "{value} in revenue over 30 days. Note any trend.",
            biz_ov,
        )

    async def _pending(self, db_type, schema, roles, db_cfg, biz_ov):
        table = roles.get("orders") or roles.get("transactions")
        if not table:
            return None
        scol = _status_col(table, schema)
        if not scol:
            return None
        sql = (
            f"SELECT COUNT(*) AS value FROM {table} "
            f"WHERE LOWER(CAST({scol} AS VARCHAR)) IN ('pending','processing','new','open','unpaid','awaiting')"
        )
        return await self._run(
            "pending_orders",
            "Pending Orders",
            "📦",
            "operations",
            sql,
            db_cfg,
            "{value} pending/open orders. High or normal?",
            biz_ov,
        )

    async def _new_customers(self, db_type, schema, roles, db_cfg, biz_ov):
        table = roles.get("customers")
        if not table:
            return None
        ccol = _created_col(table, schema)
        if ccol:
            interval = (
                "NOW() - INTERVAL '7 days'"
                if db_type == "postgresql"
                else "DATE_SUB(NOW(), INTERVAL 7 DAY)"
            )
            sql = f"SELECT COUNT(*) AS value FROM {table} WHERE {ccol} >= {interval}"
        else:
            sql = f"SELECT COUNT(*) AS value FROM {table}"
        return await self._run(
            "new_customers_7d",
            "New Customers (7d)",
            "👤",
            "growth",
            sql,
            db_cfg,
            "{value} new customers this week.",
            biz_ov,
        )

    async def _total_records(self, db_type, schema, roles, db_cfg, biz_ov):
        table = (
            roles.get("orders")
            or roles.get("customers")
            or roles.get("transactions")
            or next(iter(schema.keys()), None)
        )
        if not table:
            return None
        sql = f"SELECT COUNT(*) AS value FROM {table}"
        return await self._run(
            "total_records",
            f"Total in {table}",
            "📊",
            "operations",
            sql,
            db_cfg,
            f"Total records in {table}: {{value}}.",
            biz_ov,
        )

    async def _recent_activity(self, db_type, schema, roles, db_cfg, biz_ov):
        table = (
            roles.get("orders") or roles.get("transactions") or roles.get("customers")
        )
        if not table:
            return None
        ccol = _created_col(table, schema)
        if not ccol:
            return None
        interval = (
            "NOW() - INTERVAL '24 hours'"
            if db_type == "postgresql"
            else "DATE_SUB(NOW(), INTERVAL 1 DAY)"
        )
        sql = f"SELECT COUNT(*) AS value FROM {table} WHERE {ccol} >= {interval}"
        return await self._run(
            "activity_24h",
            "Activity (Last 24h)",
            "⚡",
            "operations",
            sql,
            db_cfg,
            "{value} records in the last 24 hours.",
            biz_ov,
        )

    async def _status_breakdown(self, db_type, schema, roles, db_cfg, biz_ov):
        table = roles.get("orders") or roles.get("transactions")
        if not table:
            return None
        scol = _status_col(table, schema)
        if not scol:
            return None
        sql = f"SELECT {scol} AS status, COUNT(*) AS count FROM {table} GROUP BY {scol} ORDER BY count DESC LIMIT 8"
        return await self._run(
            "status_breakdown",
            "Status Breakdown",
            "📈",
            "operations",
            sql,
            db_cfg,
            "Status distribution: {value}.",
            biz_ov,
            value_key="status",
        )

    async def _run(
        self,
        id,
        label,
        icon,
        category,
        sql,
        db_cfg,
        prompt_tpl,
        biz_ov,
        value_key="value",
    ):
        try:
            logger.debug(f"InsightEngine [{id}]: {sql}")
            result = await DirectSQLExecutor.execute_with_config(
                sql=sql, database_config=db_cfg
            )
            if not result.get("success"):
                logger.warning(f"InsightEngine [{id}] SQL error: {result.get('error')}")
                return None
            data = result.get("data", [])
            if not data:
                return {
                    "id": id,
                    "label": label,
                    "icon": icon,
                    "category": category,
                    "status": "no_data",
                    "value": None,
                    "narrative": "No data found for this metric.",
                    "sql_used": sql,
                    "generated_at": datetime.utcnow().isoformat(),
                }

            first = data[0]
            if value_key == "value":
                raw = (
                    first.get("value")
                    or first.get("count")
                    or first.get("total")
                    or list(first.values())[0]
                )
                try:
                    n = float(raw)
                    display = f"{n:,.0f}" if n == int(n) else f"{n:,.2f}"
                except (TypeError, ValueError):
                    display = str(raw)
            else:
                display = ", ".join(
                    f"{r.get(value_key, '?')}: {list(r.values())[-1]}" for r in data[:5]
                )

            narrative = await self._narrate(
                label, prompt_tpl.format(value=display), data, biz_ov
            )
            return {
                "id": id,
                "label": label,
                "icon": icon,
                "category": category,
                "status": "ok",
                "value": display,
                "data": data[:10],
                "narrative": narrative,
                "sql_used": sql,
                "generated_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"InsightEngine [{id}] exception: {e}", exc_info=True)
            return None

    async def _narrate(self, label, prompt, data, biz_ov):
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise business intelligence analyst. "
                        "Write exactly 1-2 sentences: the key finding + one concrete recommendation. "
                        f"Business context: {biz_ov[:300] if biz_ov else 'Unknown'}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Metric: {label}\nData: {str(data[:5])}\n{prompt}",
                },
            ]
            if self.llm._should_use_groq():
                return await self.llm._generate_with_groq(messages)
            try:
                r = await self.llm._generate_with_gemini(messages)
                self.llm.rate_limiter.add_request("gemini")
                return r
            except Exception:
                return await self.llm._generate_with_groq(messages)
        except Exception as e:
            logger.error(f"InsightEngine narrative failed: {e}")
            return f"{label} data retrieved successfully."


async def get_recent_insights_summary(
    insights: List[Dict[str, Any]], max_items: int = 3
) -> str:
    if not insights:
        return ""
    ok = [i for i in insights if i.get("status") == "ok"][:max_items]
    if not ok:
        return ""
    lines = ["RECENT BUSINESS METRICS (auto-generated):"]
    for i in ok:
        lines.append(f"• {i['icon']} {i['label']}: {i['value']} — {i['narrative']}")
    return "\n".join(lines)
