SELECT Churn, SubscriptionType, COUNT(*) AS count_type
FROM customer_churn
GROUP BY Churn, SubscriptionType
ORDER BY Churn, count_type DESC;
