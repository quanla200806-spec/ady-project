USE [master];
GO

/*
    VeraHeart ADY201m - SQL Server storage script

    Muc tieu:
    - Chi tao database VeraHeart neu database chua ton tai.
    - Chi tao bang dbo.heart_data neu bang chua ton tai.
    - Neu bang da ton tai thi khong xoa, khong tao lai, chi bo sung cot thieu.
    - Dam bao cot id co rang buoc PRIMARY KEY hoac UNIQUE de khong trung benh nhan.
*/

IF DB_ID(N'VeraHeart') IS NULL
BEGIN
    CREATE DATABASE [VeraHeart];
END
GO

USE [VeraHeart];
GO

IF OBJECT_ID(N'dbo.heart_data', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.heart_data (
        id INT NOT NULL,
        age INT NOT NULL,
        gender INT NULL,
        height INT NOT NULL,
        weight DECIMAL(10, 2) NOT NULL,
        ap_hi INT NOT NULL,
        ap_lo INT NOT NULL,
        cholesterol INT NULL,
        gluc INT NULL,
        smoke INT NULL,
        alco INT NULL,
        active INT NULL,
        cardio BIT NOT NULL,
        BMI DECIMAL(12, 8) NOT NULL,
        CONSTRAINT PK_heart_data PRIMARY KEY (id),
        CONSTRAINT CK_heart_data_cardio CHECK (cardio IN (0, 1))
    );
END
GO

/*
    Neu bang da co san, bo sung cot thieu ma khong dung DROP/CREATE lai bang.
    Cot them vao bang cu de NULL de tranh loi voi du lieu cu dang ton tai.
*/
IF COL_LENGTH(N'dbo.heart_data', N'id') IS NULL
    ALTER TABLE dbo.heart_data ADD id INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'age') IS NULL
    ALTER TABLE dbo.heart_data ADD age INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'gender') IS NULL
    ALTER TABLE dbo.heart_data ADD gender INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'height') IS NULL
    ALTER TABLE dbo.heart_data ADD height INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'weight') IS NULL
    ALTER TABLE dbo.heart_data ADD weight DECIMAL(10, 2) NULL;
IF COL_LENGTH(N'dbo.heart_data', N'ap_hi') IS NULL
    ALTER TABLE dbo.heart_data ADD ap_hi INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'ap_lo') IS NULL
    ALTER TABLE dbo.heart_data ADD ap_lo INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'cholesterol') IS NULL
    ALTER TABLE dbo.heart_data ADD cholesterol INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'gluc') IS NULL
    ALTER TABLE dbo.heart_data ADD gluc INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'smoke') IS NULL
    ALTER TABLE dbo.heart_data ADD smoke INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'alco') IS NULL
    ALTER TABLE dbo.heart_data ADD alco INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'active') IS NULL
    ALTER TABLE dbo.heart_data ADD active INT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'cardio') IS NULL
    ALTER TABLE dbo.heart_data ADD cardio BIT NULL;
IF COL_LENGTH(N'dbo.heart_data', N'BMI') IS NULL
    ALTER TABLE dbo.heart_data ADD BMI DECIMAL(12, 8) NULL;
GO

/*
    Xu ly du lieu cu bi loi truoc khi them khoa id.

    Ly do can lam:
    - Neu truoc do da append CSV nhieu lan, bang SQL co the co nhieu dong cung id.
    - Khi id bi trung thi khong the them PRIMARY KEY/UNIQUE va app khong the UPSERT an toan.

    Cach xu ly:
    - Backup toan bo bang hien tai sang dbo.heart_data_backup_before_dedup neu chua co.
    - Xoa dong id NULL vi du lieu nay khong the dong bo theo id.
    - Voi id bi trung, giu lai 1 dong va xoa cac dong lap.
*/
IF (
    EXISTS (
        SELECT 1
        FROM dbo.heart_data
        WHERE id IS NULL
    )
    OR EXISTS (
        SELECT 1
        FROM dbo.heart_data
        WHERE id IS NOT NULL
        GROUP BY id
        HAVING COUNT(*) > 1
    )
)
BEGIN
    IF OBJECT_ID(N'dbo.heart_data_backup_before_dedup', N'U') IS NULL
    BEGIN
        SELECT *
        INTO dbo.heart_data_backup_before_dedup
        FROM dbo.heart_data;
    END;

    DELETE FROM dbo.heart_data
    WHERE id IS NULL;

    WITH duplicated AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) AS rn
        FROM dbo.heart_data
    )
    DELETE FROM duplicated
    WHERE rn > 1;
END
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes i
    WHERE i.object_id = OBJECT_ID(N'dbo.heart_data')
      AND i.is_unique = 1
      AND EXISTS (
          SELECT 1
          FROM sys.index_columns ic
          JOIN sys.columns c
              ON ic.object_id = c.object_id AND ic.column_id = c.column_id
          WHERE ic.object_id = i.object_id
            AND ic.index_id = i.index_id
            AND ic.key_ordinal = 1
            AND c.name = N'id'
      )
      AND NOT EXISTS (
          SELECT 1
          FROM sys.index_columns ic
          WHERE ic.object_id = i.object_id
            AND ic.index_id = i.index_id
            AND ic.key_ordinal > 1
      )
)
BEGIN
    ALTER TABLE dbo.heart_data ALTER COLUMN id INT NOT NULL;
    ALTER TABLE dbo.heart_data
    ADD CONSTRAINT UQ_heart_data_id UNIQUE (id);
END
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_heart_data_cardio'
      AND parent_object_id = OBJECT_ID(N'dbo.heart_data')
)
BEGIN
    ALTER TABLE dbo.heart_data
    ADD CONSTRAINT CK_heart_data_cardio CHECK (cardio IN (0, 1));
END
GO

SELECT
    COUNT(*) AS total_patients
FROM dbo.heart_data;
GO
