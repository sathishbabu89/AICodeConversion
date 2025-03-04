       IDENTIFICATION DIVISION.
       PROGRAM-ID. MQ-RECEIVER.
       ENVIRONMENT DIVISION.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-MQ-MSG      PIC X(100).

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           EXEC CICS GET
               QUEUE('ORDER-QUEUE')
               INTO(WS-MQ-MSG)
               LENGTH(100)
           END-EXEC.

           DISPLAY 'Received MQ Message: ' WS-MQ-MSG.
           STOP RUN.
