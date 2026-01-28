"""
Subscription Validation Middleware for Python Engine

This validates subscription context passed from Laravel backend
and enforces rate limiting based on plan limits.
"""

from fastapi import Request, HTTPException
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SubscriptionContext:
    """
    Subscription context passed from Laravel backend
    """
    def __init__(self, data: Dict):
        self.plan_id = data.get('plan_id', 'starter')
        self.status = data.get('status', 'active')
        self.limits = data.get('limits', {})
        self.current_usage = data.get('current_usage', {})
        self.business_id = data.get('business_id')
        
    def has_feature(self, feature: str) -> bool:
        """Check if plan has access to a feature"""
        feature_plans = {
            'database_query': ['professional', 'enterprise'],
            'advanced_analytics': ['enterprise'],
            'unlimited_conversations': ['enterprise'],
        }
        
        required_plans = feature_plans.get(feature, ['starter'])
        return self.plan_id in required_plans
    
    def get_limit(self, limit_type: str) -> int:
        """Get limit for a specific type"""
        limit = self.limits.get(limit_type, 0)
        
        # -1 or 999999 means unlimited
        if limit == -1 or limit >= 999999:
            return float('inf')
        
        return limit
    
    def get_usage(self, limit_type: str) -> int:
        """Get current usage for a type"""
        return self.current_usage.get(f"{limit_type}_used", 0)
    
    def is_limit_exceeded(self, limit_type: str) -> bool:
        """Check if limit is exceeded"""
        limit = self.get_limit(limit_type)
        usage = self.get_usage(limit_type)
        
        if limit == float('inf'):
            return False
        
        return usage >= limit
    
    def is_active(self) -> bool:
        """Check if subscription is active"""
        return self.status in ['active', 'trialing']


async def get_subscription_context(request: Request) -> Optional[SubscriptionContext]:
    """
    Extract subscription context from request headers
    
    Laravel backend should send these headers:
    - X-Subscription-Plan: plan_id
    - X-Subscription-Status: status
    - X-Subscription-Limits: JSON string of limits
    - X-Subscription-Usage: JSON string of current usage
    - X-Business-Id: business_id
    """
    try:
        import json
        
        plan_id = request.headers.get('X-Subscription-Plan', 'starter')
        status = request.headers.get('X-Subscription-Status', 'active')
        business_id = request.headers.get('X-Business-Id')
        
        # Parse limits and usage from JSON headers
        limits_json = request.headers.get('X-Subscription-Limits', '{}')
        usage_json = request.headers.get('X-Subscription-Usage', '{}')
        
        limits = json.loads(limits_json)
        current_usage = json.loads(usage_json)
        
        context = SubscriptionContext({
            'plan_id': plan_id,
            'status': status,
            'limits': limits,
            'current_usage': current_usage,
            'business_id': business_id
        })
        
        return context
        
    except Exception as e:
        logger.error(f"Error parsing subscription context: {e}")
        # Return default free context on error
        return SubscriptionContext({
            'plan_id': 'starter',
            'status': 'active',
            'limits': {
                'conversations': 1000,
                'chatbots': 1,
            },
            'current_usage': {},
            'business_id': None
        })


def validate_subscription_active(context: SubscriptionContext):
    """Validate that subscription is active"""
    if not context.is_active():
        raise HTTPException(
            status_code=402,
            detail={
                'error': 'subscription_inactive',
                'message': 'Your subscription is not active',
                'status': context.status
            }
        )


def validate_feature_access(context: SubscriptionContext, feature: str):
    """Validate access to a specific feature"""
    if not context.has_feature(feature):
        raise HTTPException(
            status_code=402,
            detail={
                'error': 'feature_locked',
                'message': f'This feature requires a higher plan',
                'feature': feature,
                'current_plan': context.plan_id
            }
        )


def validate_usage_limit(context: SubscriptionContext, limit_type: str):
    """Validate usage limit"""
    if context.is_limit_exceeded(limit_type):
        raise HTTPException(
            status_code=402,
            detail={
                'error': 'limit_exceeded',
                'message': f'You have reached your {limit_type} limit',
                'limit_type': limit_type,
                'current_usage': context.get_usage(limit_type),
                'limit': context.get_limit(limit_type),
                'current_plan': context.plan_id
            }
        )


# ========================================
# Rate Limiting Based on Plan
# ========================================

class PlanBasedRateLimiter:
    """
    Rate limiter that adjusts based on subscription plan
    """
    
    # Requests per minute by plan
    RATE_LIMITS = {
        'starter': 10,      # 10 req/min
        'professional': 60,  # 60 req/min
        'enterprise': 300    # 300 req/min (virtually unlimited)
    }
    
    def __init__(self):
        self.request_counts: Dict[str, int] = {}
        
    def get_limit(self, plan_id: str) -> int:
        """Get rate limit for plan"""
        return self.RATE_LIMITS.get(plan_id, 10)
    
    def check_rate_limit(self, business_id: str, plan_id: str) -> bool:
        """
        Check if request should be rate limited
        
        Returns True if rate limit exceeded
        """
        # In production, use Redis with sliding window
        # This is a simplified example
        
        limit = self.get_limit(plan_id)
        
        # Simple counter (replace with Redis in production)
        key = f"{business_id}:{plan_id}"
        current_count = self.request_counts.get(key, 0)
        
        if current_count >= limit:
            return True
        
        self.request_counts[key] = current_count + 1
        return False


rate_limiter = PlanBasedRateLimiter()


async def validate_rate_limit(context: SubscriptionContext):
    """Validate rate limit based on plan"""
    if not context.business_id:
        return  # Skip for public requests
    
    if rate_limiter.check_rate_limit(context.business_id, context.plan_id):
        raise HTTPException(
            status_code=429,
            detail={
                'error': 'rate_limit_exceeded',
                'message': f'Rate limit exceeded for {context.plan_id} plan',
                'plan': context.plan_id,
                'limit': rate_limiter.get_limit(context.plan_id)
            }
        )