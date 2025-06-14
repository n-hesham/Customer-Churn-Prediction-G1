SELECT 
  Churn, 
  AVG(`SupportCalls`) AS avg_calls, 
  MAX(`SupportCalls`) AS max_calls
FROM `customer_churn`
GROUP BY Churn;
