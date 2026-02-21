
# Lecture 0 Introduction

- Databases (and SQL) are tools that can be used to interact with, store, and manage information. Although the tools we’re using in this course are new, a database is an age-old idea.
- Look at this diagram from a few thousand years ago. It has rows and columns, and seems to contain stipends for workers at a temple. One could call this diagram a table, or even a spreadsheet.

- Based on what we see in the diagram above, we can conclude that: 
    - A table stores some set of information (here, worker stipends).
    - Every row in a table stores one item in that set (here, one worker).
    - Every column has some attribute of that item (here, the stipend for a particular month).

Three reasons to move beyond spreadsheets to databases are
1. Scale: Databases can store not just items numbering to tens of thousands but even millions and billions.
2. Update Capacity: Databases are able to handle multiple updates of data in a second.
3. Speed: Databases allow faster look-up of information. This is because databases provide us with access to different algorithms to retrieve information. In contrast, spreadsheets that merely allow the use of Ctrl+F or Cmd+F to go through hits one at a time.

# What is a Database?
- A database is a way of organizing data such that you can perform four operations on it
    - create
    - read
    - update
    - delete
- A database management system (DBMS) is a way to interact with a database using a graphical interface or textual language.
- Examples of DBMS: MySQL, Oracle, PostgreSQL, SQLite, Microsoft Access, MongoDB etc.


# SELECT

This selects all the rows from the table called `longlist`.


- In the SQLite environment, run
```sql
SELECT * 
FROM "longlist";
```
This selects all the rows from the table called `longlist`.

- The output we get contains all the columns of all the rows in this table, which is a lot of data. We can simplify it by selecting a particular column, say the title, from the table. Let’s try
```sql
SELECT "title" 
FROM "longlist";
```
- Now, we see a list of the titles in this table. But what if we want to see titles and authors in our search results? For this, we run
```sql
SELECT "title", "author" 
FROM longlist;
```



# LIMIT

- If a database had millions of rows, it might not make sense to select all of its rows. Instead, we might want to merely take a peek at the data it contains. We use the SQL keyword `LIMIT` to specify the number of rows in the query output.

```sql
SELECT "title" 
FROM "longlist" 
LIMIT 10;
```
- This query gives us the first 10 titles in the database. The titles are ordered the same way in the output of this query as they are in the database.

## WHERE

- The keyword `WHERE` is used to select rows based on a condition; it will output the rows for which the specified condition is true.
- 
```sql
SELECT "title", "author" 
FROM "longlist" 
WHERE "year" = 2023;
```
This gives us the titles and authors for the books longlisted in 2023. Note that `2023` is not in quotes because it is an integer, not a string or identifier.

- The operators that can be used to specify conditions in SQL are `=` (“equal to”), `!=` (“not equal to”) and `<>` (also “not equal to”).

- To select the books that are not hardcovers, we can run the query
```sql
SELECT "title", "format" 
FROM "longlist" 
WHERE "format" != 'hardcover';
```
Note that hardcover is in single quotes because it is an SQL string and not an identifier.
`!=` can be replaced with the operator `<>` to get the same results. The modified query would be
```sql
SELECT "title", "format" 
FROM "longlist" 
WHERE "format" <> 'hardcover';
```
Yet another way to get the same results is to use the SQL keyword `NOT`. The modified query would be
```sql 
SELECT "title", "format" 
FROM "longlist" 
WHERE NOT "format" = 'hardcover';
```
- To combine conditions, we can use the SQL keywords `AND` and `OR`. We can also use parentheses to indicate how to combine the conditions in a compound conditional statement.
- To select the titles and authors of the books longlisted in 2022 or 2023
```sql
SELECT "title", "author" 
FROM "longlist" 
WHERE "year" = 2022 OR "year" = 2023;
```
To select the books longlisted in 2022 or 2023 that were not hardcovers
```sql
SELECT "title", "format" 
FROM "longlist" 
WHERE ("year" = 2022 OR "year" = 2023) AND "format" != 'hardcover';
```
Here, the parantheses indicate that the `OR` clause should be evaluated before the `AND` clause.

## NULL 

- It is possible that tables may have missing data. `NULL` is a type used to indicate that certain data does not have a value, or does not exist in the table.
- For example, the books in our database have a translator along with an author. However, only some of the books have been translated to English. For other books, the translator value will be `NULL`.
- Conditions used with `NULL` are `IS NULL` and `IS NOT NULL`.
- To select the books for which translators don’t exist, we can run
```sql
SELECT "title", "translator" 
FROM "longlist"
WHERE "translator" IS NULL;
```
- Let’s try the reverse: selecting the books for which translators do exist.
```sql
SELECT "title", "translator" 
FROM "longlist"
WHERE "translator" IS NOT NULL;
```

## LIKE

- This keyword is used to select data that roughly matches the specified string. For example, `LIKE` could be used to select books that have a certain word or phrase in their title.
- `LIKE` is combined with the operators `%` (matches any characters around a given string) and `_` (matches a single character).
- To select the books with the word “love” in their titles, we can run
```sql
SELECT "title"
FROM "longlist"
WHERE "title" LIKE '%love%';
```
`%` matches 0 or more characters, so this query would match book titles that have 0 or more characters before and after “love” — that is, titles that contain “love”.

- To select the books whose title begin with “The”, we can run
```sql
SELECT "title" 
FROM "longlist" 
WHERE "title" LIKE 'The%';
```
- The above query may also return books whose titles begin with “Their” or “They”. To select only the books whose titles begin with the word “The”, we can add a space.
```sql
SELECT "title" 
FROM "longlist" 
WHERE "title" LIKE 'The %';
```
- Given that there is a book in the table whose name is either “Pyre” or “Pire”, we can select it by running
```sql
SELECT "title" 
FROM "longlist" 
WHERE "title" LIKE 'P_re';
```
- This query could also return book titles like “Pore” or “Pure” if they existed in our database, because `_` matches any single character.

## Ranges

- We can also use the operators `<`,`>`, `<=` and `>=` in our conditions to match a range of values. For example, to select all the books longlisted between the years 2019 and 2022 (inclusive), we can run
```sql
SELECT "title", "author" 
FROM "longlist" 
WHERE "year" >= 2019 AND "year" <= 2022;
```
- Another way to get the same results is using the keywords BETWEEN and AND to specify inclusive ranges. We can run
```sql
SELECT "title", "author" 
FROM "longlist" 
WHERE "year" BETWEEN 2019 AND 2022;
```
- To select the books that have a rating of 4.0 or higher, we can run
```sql
SELECT "title", "rating" 
FROM "longlist" 
WHERE "rating" > 4.0;
```
- To further limit the selected books by number of votes, and have only those books with at least 10,000 votes, we can run
```sql
SELECT "title", "rating", "votes" 
FROM "longlist" 
WHERE "rating" > 4.0 AND "votes" > 10000;
```
- To select the books that have less than 300 pages, we can run
```sql 
SELECT "title", "pages" 
FROM "longlist" 
WHERE "pages" < 300;
```

## ORDER BY

- The `ORDER BY` keyword allows us to organize the returned rows in some specified order.
- The following query selects the bottom 10 books in our database by rating.
```sql
SELECT "title", "rating" 
FROM "longlist" 
ORDER BY "rating" LIMIT 10;
```
- Note that we get the bottom 10 books because ORDER BY chooses ascending order by default.
- Instead, to select the top 10 books
```sql
SELECT "title", "rating" 
FROM "longlist" 
ORDER BY "rating" DESC LIMIT 10;
```
- Note the use of the SQL keyword `DESC` to specify the descending order. `ASC` can be used to explicitly specify ascending order.

- To select the top 10 books by rating and also include number of votes as a tie-break, we can run
```sql 
SELECT "title", "rating", "votes" 
FROM "longlist"
ORDER BY "rating" DESC, "votes" DESC 
LIMIT 10;
```
- Note that for each column in the `ORDER BY` clause, we specify ascending or descending order.

## Questions

1. To sort books by title alphabetically, can we use `ORDER BY`?

## Aggregate Functions


- `COUNT`, `AVG`, `MIN`, `MAX`, and `SUM` are called aggregate functions and allow us to perform the corresponding operations over multiple rows of data. By their very nature, each of the following aggregate functions will return only a single output—the aggregated value.
- To find the average rating of all books in the database
```sql
SELECT AVG("rating") 
FROM "longlist";
```
- To round the average rating to 2 decimal points
```sql
SELECT ROUND(AVG("rating"), 2) 
FROM "longlist";
```
- To rename the column in which the results are displayed
```sql
SELECT ROUND(AVG("rating"), 2) AS "average rating" 
FROM "longlist";
```
Note the use of the SQL keyword AS to rename columns.

- To select the maximum rating in the database
```sql
SELECT MAX("rating") 
FROM "longlist";
```
- To select the minimum rating in the database
```sql 
SELECT MIN("rating") 
FROM "longlist";
```
- To count the total number of votes in the database
```sql 
SELECT SUM("votes") 
FROM "longlist";
```
- To count the number of books in our database
```sql
SELECT COUNT(*) 
FROM "longlist";
```
- Remember that we used * to select every row and column from the database. In this case, we are trying to count every row in the database and hence we use the *.
- To count the number of translators
```sql
SELECT COUNT("translator") 
FROM "longlist";
```
- We observe that the number of translators is fewer than the number of rows in the database. This is because the `COUNT` function does not count `NULL` values.
- To count the number of publishers in the database
```sql
SELECT COUNT("publisher") 
FROM "longlist";
```
- As with translators, this query will count the number of publisher values that are not `NULL`. However, this may include duplicates. Another SQL keyword, `DISTINCT`, can be used to ensure that only distinct values are counted.
```sql
SELECT COUNT(DISTINCT "publisher") 
FROM "longlist";
```