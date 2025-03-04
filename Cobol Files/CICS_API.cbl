       IDENTIFICATION DIVISION.
       PROGRAM-ID. GET-CUSTOMER.
       ENVIRONMENT DIVISION.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-CUST-ID          PIC X(10).
       01  WS-CUST-NAME        PIC X(50).
       01  WS-RESPONSE         PIC X(100).

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           DISPLAY 'Enter Customer ID: '.
           ACCEPT WS-CUST-ID.

           EXEC CICS READ
               DATASET('CUSTOMER-FILE')
               INTO(WS-CUST-NAME)
               RIDFLD(WS-CUST-ID)
               END-EXEC.

           MOVE 'Customer Name: ' TO WS-RESPONSE.
           STRING WS-RESPONSE WS-CUST-NAME DELIMITED BY SIZE INTO WS-RESPONSE.
           DISPLAY WS-RESPONSE.
           STOP RUN.
