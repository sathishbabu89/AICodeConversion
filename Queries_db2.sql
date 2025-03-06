-- Query 1: Select all employees in a specific department
SELECT e.employee_id, e.first_name, e.last_name, e.salary
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE d.department_name = 'Sales';

-- Query 2: Find the total budget for all projects
SELECT SUM(budget) AS total_budget
FROM projects;

-- Query 3: Insert a new employee
INSERT INTO employees (employee_id, first_name, last_name, hire_date, salary, department_id)
VALUES (101, 'John', 'Doe', '2023-01-15', 75000.00, 1);

-- Query 4: Update employee salary
UPDATE employees
SET salary = salary * 1.10
WHERE department_id = 2;

-- Query 5: Delete a project
DELETE FROM projects
WHERE project_id = 5;
