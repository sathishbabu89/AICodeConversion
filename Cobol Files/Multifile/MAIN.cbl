       IDENTIFICATION DIVISION.
       PROGRAM-ID. MAIN.
       
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       COPY "COMMON.cpy".

       PROCEDURE DIVISION.
           DISPLAY "Executing MAIN Program".
           CALL 'UTILITY' USING WS-VALUE.
           STOP RUN.
       END PROGRAM MAIN.
