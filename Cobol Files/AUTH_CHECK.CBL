       IDENTIFICATION DIVISION.
       PROGRAM-ID. LOGIN.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-USERNAME       PIC X(20).
       01 WS-PASSWORD       PIC X(20).
       01 WS-VALID-USER     PIC X(3) VALUE 'NO'.

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           DISPLAY 'Enter Username: '.
           ACCEPT WS-USERNAME.
           DISPLAY 'Enter Password: '.
           ACCEPT WS-PASSWORD.

           IF WS-USERNAME = 'ADMIN' AND WS-PASSWORD = 'SECRET'
               MOVE 'YES' TO WS-VALID-USER
               DISPLAY 'Login Successful'
           ELSE
               DISPLAY 'Invalid Credentials'.
           STOP RUN.
