IDENTIFICATION DIVISION.
PROGRAM-ID. MQSEND.

WORKING-STORAGE SECTION.
01  WS-MSG          PIC X(100) VALUE 'Hello from COBOL to MQ!'.
01  WS-QNAME        PIC X(20) VALUE 'QUEUE1'.

PROCEDURE DIVISION.
    EXEC CICS LINK PROGRAM('MQPUT')
        COMMAREA(WS-MSG)
    END-EXEC.

    DISPLAY "Message sent to MQ Queue: " WS-QNAME.

    STOP RUN.
