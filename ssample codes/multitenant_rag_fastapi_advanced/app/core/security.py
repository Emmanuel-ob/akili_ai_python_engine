
from fastapi import Header, HTTPException

class TenantContext:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

def get_tenant_context(x_tenant_token: str = Header(...)):
    if not x_tenant_token:
        raise HTTPException(status_code=401)
    return TenantContext(x_tenant_token)
