SELECT Churn, 
       AVG(`PaymentDelay`) AS avg_delay, 
       MAX(`PaymentDelay`) AS max_delay
FROM customer_churn
GROUP BY Churn;
