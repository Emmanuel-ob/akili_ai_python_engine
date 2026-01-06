
from sqlalchemy import create_engine

TENANT_DATABASES = {
    "acme": "postgresql://readonly:password@localhost/acme"
}

TENANT_SCHEMAS = {
    "acme": {
        "orders": {
            "description": "Customer orders",
            "columns": {
                "id": "order id",
                "status": "order status",
                "created_at": "creation timestamp"
            }
        }
    }
}

def get_engine(tenant_id: str):
    url = TENANT_DATABASES.get(tenant_id)
    if not url:
        raise ValueError("Unknown tenant")
    return create_engine(url)

def get_schema(tenant_id: str):
    schema = TENANT_SCHEMAS.get(tenant_id)
    if not schema:
        raise ValueError("No schema found for tenant")
    return schema
