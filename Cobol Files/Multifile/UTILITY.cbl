       IDENTIFICATION DIVISION.
       PROGRAM-ID. UTILITY.
       
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       COPY "COMMON.cpy".
       
       PROCEDURE DIVISION.
           DISPLAY "Executing UTILITY Program".
           MOVE "UTILITY DONE" TO WS-VALUE.
           STOP RUN.
       END PROGRAM UTILITY.
