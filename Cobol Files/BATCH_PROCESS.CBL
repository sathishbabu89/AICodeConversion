       IDENTIFICATION DIVISION.
       PROGRAM-ID. FILE-PROCESSOR.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT IN-FILE ASSIGN TO 'INPUT.DAT'
           ORGANIZATION IS LINE SEQUENTIAL.
           SELECT OUT-FILE ASSIGN TO 'OUTPUT.DAT'
           ORGANIZATION IS LINE SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.
       FD IN-FILE.
       01 IN-RECORD.
           05 IN-NAME PIC X(50).
           05 IN-AGE  PIC 9(2).

       FD OUT-FILE.
       01 OUT-RECORD.
           05 OUT-NAME PIC X(50).
           05 OUT-MESSAGE PIC X(30).

       WORKING-STORAGE SECTION.
       01 WS-END-OF-FILE PIC X(3) VALUE 'NO'.

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           OPEN INPUT IN-FILE.
           OPEN OUTPUT OUT-FILE.

           PERFORM UNTIL WS-END-OF-FILE = 'YES'
               READ IN-FILE INTO IN-RECORD
                   AT END MOVE 'YES' TO WS-END-OF-FILE
                   NOT AT END
                       MOVE IN-NAME TO OUT-NAME
                       MOVE 'Processed Successfully' TO OUT-MESSAGE
                       WRITE OUT-RECORD
               END-READ
           END-PERFORM.

           CLOSE IN-FILE.
           CLOSE OUT-FILE.
           STOP RUN.
