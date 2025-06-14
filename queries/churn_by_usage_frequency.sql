SELECT Gender, 
       COUNT(*) AS total, 
       SUM(Churn) AS churned, 
       ROUND(SUM(Churn)/COUNT(*) * 100, 2) AS churn_rate
FROM customer_churn
GROUP BY Gender;
