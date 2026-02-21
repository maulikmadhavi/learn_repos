-- DEsc: This is lecture from CS50 on querying
-- Title: Querying 

-- Selecting all rows from a table
SELECT * FROM longlist;

-- selecting limited number of rows
SELECT * FROM  "longlist" LIMIT 10;

-- Selecting specific columns where the condition is met
SELECT "title" FROM "longlist" WHERE "year" = 2023; 

-- Template of a given table
.schema "longlist"

-- Logical Operators: AND, OR, NOT
SELECT "title","year","rating" FROM "longlist" WHERE "year" = 2023 OR "rating" > 4.5;

-- Get the NULL values
SELECT "title","year","rating" FROM "longlist" WHERE "translator" IS NULL;

-- Fuzzy search using LIKE
SELECT "title","year","rating" FROM "longlist" WHERE "title" LIKE "%the%";

SELECT "title","year","rating" FROM "longlist" WHERE "title" LIKE "%b__k%";

-- Range based search
SELECT "title","year","rating" FROM "longlist" WHERE "year" BETWEEN 2020 AND 2023;

SELECT "title","year","rating" FROM "longlist" WHERE "rating" BETWEEN 4.0 AND 4.5;

-- Sorting the results
SELECT "title","year","rating" FROM "longlist" WHERE "year" = 2023 ORDER BY "rating" DESC;

SELECT "title","year","rating" FROM "longlist" WHERE "rating" > 3.5 ORDER BY "rating" DESC, "year" ASC;

-- Aggregating the results

-- COUNT, SUM, AVG, MIN, MAX, DISTINCT

SELECT COUNT(*) FROM "longlist";

SELECT COUNT(DISTINCT "year") FROM "longlist";

SELECT ROUND(AVG("rating"),3) AS 'rate' FROM "longlist";

SELECT DISTINCT("author") FROM "longlist";
