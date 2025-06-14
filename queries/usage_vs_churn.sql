SELECT 
  Churn, 
  AVG(`UsageFrequency`) AS avg_usage, 
  STDDEV(`UsageFrequency`) AS stddev_usage
FROM `customer_churn`
GROUP BY Churn;
