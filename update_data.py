import json
import re

file_path = "frontend/src/data.ts"
with open(file_path, "r") as f:
    content = f.read()

# For hotel bookings
hotel_insights = """[
      { title: 'Solo travelers cancel frequently', details: 'Solo travelers show highest cancellation rate, mostly associated with City Hotels.', metric: '34% risk', impact: 'High' },
      { title: 'City hotel cancellations', details: 'City hotels experience significantly more cancellations (32.1%) than resort hotels (19.4%).', metric: '32.1% rate', impact: 'Medium' },
      { title: 'Lead-time correlation', details: 'Customers booking far in advance (>120 days) cancel more frequently than short-term bookings.', metric: '82% risk increase', impact: 'High' },
      { title: 'Repeat Guest Loyalty', details: 'Returning guests have a cancellation probability of less than 3.5%, proving high loyalty.', metric: '3.5% cancellation', impact: 'Low' },
    ]"""

hotel_recommendations = """[
      { action: 'Offer retention discounts', impact: 'Critical', expectedOutcome: 'Enforce non-refundable deposit terms for high-risk lead times. Expected cancellation reduction of 15%.' },
      { action: 'Focus premium services on loyalists', impact: 'Medium', expectedOutcome: 'Implement flexible cancellation waiver options exclusively for verified repeat guests.' },
      { action: 'Optimize seasonal overbooking', impact: 'High', expectedOutcome: 'Dynamically scale down overbooking margins to 4.2% during peak months to prevent displacement.' },
    ]"""

hotel_business_kpis = """businessKPIs: [
      { id: 'rev', label: 'Total Revenue', value: '$4.2M', trend: '+12% vs last month', trendUp: true, status: 'good' },
      { id: 'canc', label: 'Cancellation Rate', value: '27.5%', trend: 'High risk segment', trendUp: false, status: 'critical' },
      { id: 'adr', label: 'Avg Daily Rate', value: '$104.5', trend: 'Stable', trendUp: true, status: 'good' },
      { id: 'loyalty', label: 'Repeat Guests', value: '3.1%', trend: 'Opportunity for growth', trendUp: true, status: 'warning' },
      { id: 'lead', label: 'Avg Lead Time', value: '104 days', trend: 'High booking window', trendUp: true, status: 'neutral' },
    ],"""

# SAAS Risk
saas_insights = """[
      { title: 'Support SLA Bottlenecks', details: 'Accounts experiencing 3 or more Priority-1 unresolved support tickets show an 88% probability of cancellation.', metric: '88% Churn Risk', impact: 'Critical' },
      { title: 'Onboarding Slip Window', details: 'New signups accessing the workspace less than 5 times in Month 1 suffer a massive 72% churn multiplier.', metric: '72% Onboarding Risk', impact: 'High' },
      { title: 'Contract Security Lift', details: 'Annual and multi-year contract holders demonstrate high systemic stability.', metric: '2.2% Contract Risk', impact: 'Low' },
    ]"""

saas_recommendations = """[
      { action: 'Automated CS Workflows', impact: 'Critical', expectedOutcome: 'Hook internal triggers to alert Customer Success managers instantly when an account drops below 5 logins in their first month.' },
      { action: 'Re-engineer App Workspace Tour', impact: 'High', expectedOutcome: 'Refactor the product activation flow to guide users to create their first agent project within 3 minutes of sign up.' },
    ]"""

saas_business_kpis = """businessKPIs: [
      { id: 'arr', label: 'Annual Rec. Revenue', value: '$12.4M', trend: '+5.2% vs last quarter', trendUp: true, status: 'good' },
      { id: 'churn', label: 'Churn Rate', value: '4.8%', trend: 'Above target (3%)', trendUp: false, status: 'critical' },
      { id: 'arpu', label: 'ARPU', value: '$840', trend: 'Stable', trendUp: true, status: 'good' },
      { id: 'health', label: 'Avg Health Score', value: '72/100', trend: 'Needs improvement', trendUp: false, status: 'warning' },
    ],"""


# ECOM
ecom_insights = """[
      { title: 'Mobile Latency Friction', details: 'Mobile users suffer a 45% abandonment spike when the checkout screen loads longer than 3.4 seconds.', metric: '45% abandonment hike', impact: 'Critical' },
      { title: 'Shipping Charge Threshold', details: 'Cart abandonment decreases by 24% when free-shipping coupons are activated for basket totals exceeding $75.', metric: '24% conversion uplift', impact: 'High' },
      { title: 'Electronics Segment Leverage', details: 'Electronics account for 42.1% of total revenue. Average order sizes are high.', metric: '42.1% Revenue', impact: 'Medium' },
    ]"""

ecom_recommendations = """[
      { action: 'Optimize Mobile Checkout', impact: 'Critical', expectedOutcome: 'Standardize responsive media files and minify analytics tags to ensure mobile checkout loads under 1.5 seconds.' },
      { action: 'Dynamic Abandonment Coupons', impact: 'High', expectedOutcome: 'Trigger an exit-intent recovery email with a custom 10% coupon code if a cart is idle for over 2 hours.' },
    ]"""

ecom_business_kpis = """businessKPIs: [
      { id: 'rev', label: 'Gross Sales', value: '$3.8M', trend: '+15% Q/Q', trendUp: true, status: 'good' },
      { id: 'conv', label: 'Conversion Rate', value: '2.4%', trend: 'Slight dip', trendUp: false, status: 'warning' },
      { id: 'aov', label: 'Avg Order Value', value: '$124', trend: 'Up from $112', trendUp: true, status: 'good' },
      { id: 'abandon', label: 'Cart Abandonment', value: '68%', trend: 'High risk area', trendUp: false, status: 'critical' },
    ],"""


# Charts replacements using regex to inject takeaway
def add_takeaways(content):
    content = re.sub(r"(title: 'Trend Analysis — ADR \(Average Daily Rate\)',)", r"\1\n        takeaway: 'ADR peaks significantly during summer months, especially in August.',", content)
    content = re.sub(r"(title: 'Distribution — Market Segment',)", r"\1\n        takeaway: 'Online Travel Agencies (Online TA) represent the vast majority of all bookings.',", content)
    content = re.sub(r"(title: 'Category Breakdown — Segment Type',)", r"\1\n        takeaway: 'Transient guests dominate the hotel volume.',", content)
    content = re.sub(r"(title: 'Correlation Matrix',)", r"\1\n        takeaway: 'Strong correlations detected between features.',", content)
    
    content = re.sub(r"(title: 'Trend Analysis — Monthly Charges',)", r"\1\n        takeaway: 'Revenue spikes strongly correlate with annual renewal periods.',", content)
    content = re.sub(r"(title: 'Distribution — Contract Duration',)", r"\1\n        takeaway: 'Month-to-month contracts constitute the bulk of the customer base.',", content)
    content = re.sub(r"(title: 'Category Breakdown — Plan Tier',)", r"\1\n        takeaway: 'Enterprise plan brings the most high-value clients.',", content)
    
    content = re.sub(r"(title: 'Trend Analysis — Price \(USD\)',)", r"\1\n        takeaway: 'Sales volume spikes during Q4 holiday events.',", content)
    content = re.sub(r"(title: 'Distribution — Product Category',)", r"\1\n        takeaway: 'Electronics drive the highest revenue volume.',", content)
    content = re.sub(r"(title: 'Category Breakdown — Device Type',)", r"\1\n        takeaway: 'Mobile users account for over 60% of all sessions.',", content)

    # Make colors more semantic (pie charts)
    content = re.sub(r"color: '#8b5cf6'", r"color: '#10b981'", content) # Green
    content = re.sub(r"color: '#3b82f6'", r"color: '#3b82f6'", content) # Blue
    content = re.sub(r"color: '#10b981'", r"color: '#f59e0b'", content) # Amber
    content = re.sub(r"color: '#f59e0b'", r"color: '#ef4444'", content) # Red

    return content


content = re.sub(r"insights: \[\n.*?\n    \],", "insights: " + hotel_insights + ",", content, flags=re.DOTALL, count=1)
content = re.sub(r"recommendations: \[\n.*?\n    \],", "recommendations: " + hotel_recommendations + ",", content, flags=re.DOTALL, count=1)
content = re.sub(r"qualityScore: 94,\n    },", "qualityScore: 94,\n    },\n    " + hotel_business_kpis, content, count=1)

content = re.sub(r"insights: \[\n.*?\n    \],", "insights: " + saas_insights + ",", content, flags=re.DOTALL, count=1)
content = re.sub(r"recommendations: \[\n.*?\n    \],", "recommendations: " + saas_recommendations + ",", content, flags=re.DOTALL, count=1)
content = re.sub(r"qualityScore: 96,\n    },", "qualityScore: 96,\n    },\n    " + saas_business_kpis, content, count=1)

content = re.sub(r"insights: \[\n.*?\n    \],", "insights: " + ecom_insights + ",", content, flags=re.DOTALL, count=1)
content = re.sub(r"recommendations: \[\n.*?\n    \],", "recommendations: " + ecom_recommendations + ",", content, flags=re.DOTALL, count=1)
content = re.sub(r"qualityScore: 88,\n    },", "qualityScore: 88,\n    },\n    " + ecom_business_kpis, content, count=1)

content = add_takeaways(content)

with open(file_path, "w") as f:
    f.write(content)

print("data.ts updated successfully")
