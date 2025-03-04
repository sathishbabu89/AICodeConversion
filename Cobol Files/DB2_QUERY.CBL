       IDENTIFICATION DIVISION.
       PROGRAM-ID. CUSTOMER-LOOKUP.
       ENVIRONMENT DIVISION.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       EXEC SQL INCLUDE SQLCA END-EXEC.

       01 WS-CUST-ID          PIC X(10).
       01 WS-CUST-NAME        PIC X(50).

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           DISPLAY 'Enter Customer ID: '.
           ACCEPT WS-CUST-ID.

           EXEC SQL
               SELECT NAME INTO :WS-CUST-NAME
               FROM CUSTOMER_TABLE
               WHERE ID = :WS-CUST-ID
           END-EXEC.

           DISPLAY 'Customer Name: ' WS-CUST-NAME.
           STOP RUN.
