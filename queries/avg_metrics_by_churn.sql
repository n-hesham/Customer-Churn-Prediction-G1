SELECT 
    Churn,
    AVG(Age) as AvgAge,
    AVG(Tenure) as AvgTenure,
    AVG(UsageFrequency) as AvgUsageFrequency,
    AVG(SupportCalls) as AvgSupportCalls,
    AVG(PaymentDelay) as AvgPaymentDelay,
    AVG(TotalSpend) as AvgTotalSpend,
    AVG(LastInteraction) as AvgLastInteraction
FROM customer_churn
GROUP BY Churn;