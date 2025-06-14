SELECT
  CASE
    WHEN Age < 25 THEN '<25'
    WHEN Age BETWEEN 25 AND 40 THEN '25-40'
    WHEN Age BETWEEN 41 AND 60 THEN '41-60'
    ELSE '60+'
  END AS age_group,
  COUNT(*) AS total
FROM customer_churn
GROUP BY age_group;
