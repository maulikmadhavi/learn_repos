



## Comprehensive SQL Commands Cheat Sheet

This cheat sheet provides a quick reference to essential SQL commands, covering basic to advanced operations. 

---

# **1. Data Query Language (DQL)**

- **SELECT**: Retrieve data from one or more tables.
  ```sql
  SELECT column1, column2 FROM table_name;
  ```

-  **SELECT** statement with a **LIMIT** clause
```sql
SELECT column1, column2
FROM table_name
LIMIT number;
```

- **WHERE**: Filter records based on conditions.
  ```sql
  SELECT column1, column2 FROM table_name WHERE condition;
  ```

- **ORDER BY**: Sort the result set.
  ```sql
  SELECT column1, column2 FROM table_name ORDER BY column1 ASC|DESC;
  ```

- **GROUP BY**: Group rows sharing a property.
  ```sql
  SELECT column1, COUNT(*) FROM table_name GROUP BY column1;
  ```

- **DISTINCT**: Return unique values.
  ```sql
  SELECT DISTINCT column_name FROM table_name;
  ```

- **HAVING**: Filter groups based on conditions.
  ```sql
  SELECT column1, COUNT(*) FROM table_name GROUP BY column1 HAVING COUNT(*) > value;
  ```

---

# **2. Data Manipulation Language (DML)**

- **INSERT INTO**: Add new records to a table.
  ```sql
  INSERT INTO table_name (column1, column2) VALUES (value1, value2);
  ```

- **UPDATE**: Modify existing records.
  ```sql
  UPDATE table_name SET column1 = value1 WHERE condition;
  ```

- **DELETE**: Remove records from a table.
  ```sql
  DELETE FROM table_name WHERE condition;
  ```

---

# **3. Data Definition Language (DDL)**

- **CREATE TABLE**: Create a new table.
  ```sql
  CREATE TABLE table_name (
      column1 datatype,
      column2 datatype,
      ...
  );
  ```

- **CREATE TABLE** statement with a **PRIMARY KEY** constraint
    ```sql
    CREATE TABLE table_name (
    column1 datatype,
    column2 datatype,
    ...
    columnN datatype,
    PRIMARY KEY (column1, column2, ...)
    );
    ```  

- **ALTER TABLE**: Modify an existing table.
  ```sql
  ALTER TABLE table_name ADD column_name datatype;
  ALTER TABLE table_name DROP COLUMN column_name;
  ```

- **DROP TABLE**: Delete a table and its data.
  ```sql
  DROP TABLE table_name;
  ```

---

# **4. Data Control Language (DCL)**

- **GRANT**: Provide user access privileges.
  ```sql
  GRANT SELECT, INSERT ON table_name TO user_name;
  ```

- **REVOKE**: Remove user access privileges.
  ```sql
  REVOKE SELECT, INSERT ON table_name FROM user_name;
  ```

---

# **5. Joins**

- **INNER JOIN**: Return records with matching values in both tables.
  ```sql
  SELECT columns FROM table1 INNER JOIN table2 ON table1.column = table2.column;
  ```

- **LEFT JOIN**: Return all records from the left table and matched records from the right table.
  ```sql
  SELECT columns FROM table1 LEFT JOIN table2 ON table1.column = table2.column;
  ```

- **RIGHT JOIN**: Return all records from the right table and matched records from the left table.
  ```sql
  SELECT columns FROM table1 RIGHT JOIN table2 ON table1.column = table2.column;
  ```

- **FULL OUTER JOIN**: Return all records when there is a match in either left or right table records.
  ```sql
  SELECT columns FROM table1 FULL OUTER JOIN table2 ON table1.column = table2.column;
  ```

---

# **6. Aggregate Functions**

- **COUNT()**: Count the number of rows.
  ```sql
  SELECT COUNT(column_name) FROM table_name;
  ```

- **SUM()**: Calculate the total sum of a numeric column.
  ```sql
  SELECT SUM(column_name) FROM table_name;
  ```

- **AVG()**: Calculate the average value of a numeric column.
  ```sql
  SELECT AVG(column_name) FROM table_name;
  ```

- **MAX()**: Find the maximum value.
  ```sql
  SELECT MAX(column_name) FROM table_name;
  ```

- **MIN()**: Find the minimum value.
  ```sql
  SELECT MIN(column_name) FROM table_name;
  ```

---

# **7. Subqueries**

- **Subquery**: A query nested inside another query.
  ```sql
  SELECT column1 FROM table_name WHERE column2 = (SELECT column2 FROM table_name WHERE condition);
  ```

---

# **8. Common SQL Clauses**

- **LIMIT**: Specify the number of records to return.
  ```sql
  SELECT column1 FROM table_name LIMIT number;
  ```

- **OFFSET**: Skip a specified number of records.
  ```sql
  SELECT column1 FROM table_name LIMIT number OFFSET number;
  ```

