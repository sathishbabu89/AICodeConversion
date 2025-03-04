IDENTIFICATION DIVISION.
PROGRAM-ID. CICSTRAN.

WORKING-STORAGE SECTION.
01  WS-IN-DATA    PIC X(20).
01  WS-OUT-DATA   PIC X(20).

PROCEDURE DIVISION.
    EXEC CICS RECEIVE
        INTO(WS-IN-DATA)
    END-EXEC.

    MOVE WS-IN-DATA TO WS-OUT-DATA.

    EXEC CICS SEND
        FROM(WS-OUT-DATA)
    END-EXEC.

    EXEC CICS RETURN END-EXEC.
