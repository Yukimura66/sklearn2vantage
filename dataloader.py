from sqlalchemy import create_engine, Integer, Float, String
import sqlalchemy
import pandas as pd
import paramiko
import scp
import time
import os
import re


def tableIsExist(tablename: str, dbname: str,
                 engine: sqlalchemy.engine.base.Engine) -> bool:
    table_list = pd.read_sql_query(
        """select * from dbc.TablesV
        where TableName = '{tablename}'
        and DatabaseName = '{dbname}'""".format(
            dbname=dbname, tablename=tablename), engine)
    return True if len(table_list) > 0 else False


def dropIfExists(tablename: str, dbname: str,
                 engine: sqlalchemy.engine.base.Engine) -> None:
    if tableIsExist(tablename, dbname, engine):
        engine.execute(f"drop table {dbname}.{tablename}")
        print(f"droped table {dbname}.{tablename}")


def dtypeParser(series: pd.Series) -> str:
    """function to change pandas data type to Teradata data type"""
    if series.dtype.name == "int64":
        parsed_dtype = "Integer"
    elif series.dtype.name == "float64":
        parsed_dtype = "float"
    else:
        max_length = max(1, max(series.astype(str).str.len())
                         )  # at least 1 length
        parsed_dtype = f"varchar({max_length})"
    return parsed_dtype


def renameColumns(series: pd.Series) -> pd.Series():
    # replace spaces to underscore, delete symbols,
    # add "col_" if name starts with digit,
    # and change "index" to "idx", which comes from reset_index
    res = (series.astype(str).str
           .replace(r"\s", "_", regex=True)
           .replace(r"\W", "", regex=True)
           .replace(r"^(\d)", "col_\\1", regex=True)
           .replace("index", "idx"))
    return res


def createTable(df: pd.DataFrame,
                engine: sqlalchemy.engine.base.Engine,
                tablename: str, dbname: str = None,
                overwrite: bool = False, indexList: list = None,
                isIndexUnique: bool = True) -> None:
    # make query string
    if dbname is None:
        dbname = engine.url.database
    new_names = renameColumns(df.columns.to_series())
    col_dtype = [dtypeParser(col) for _, col in df.iteritems()]
    column_query_part = "\n    ,".join(
        [f"{name} {dtype}" for name, dtype
         in zip(new_names, col_dtype)]
    )
    if indexList:
        idxSeries = pd.Series(indexList).astype(str)
        idx_str = ",".join(renameColumns(idxSeries))
        q_unique = "unique " if isIndexUnique else ""
        index_query_part = q_unique + f"primary index ({idx_str})"
    else:
        index_query_part = "no primary index"
    query = """
    create table {dbname}.{tablename} (
        {column_query_part}
    ) {index_query_part}
    """.format(tablename=tablename, dbname=dbname,
               column_query_part=column_query_part,
               index_query_part=index_query_part)

    # execute query
    if overwrite:
        dropIfExists(tablename, dbname, engine)
    else:
        if tableIsExist(tablename, dbname, engine):
            raise ValueError(f"{tablename}.{dbname} is already exists.")
    engine.execute(query)
    print("created table \n" + query)


def connectSSH(hostname: str, username: str, password: str = None,
               port: int = 22, keyPath: str = None
               ) -> paramiko.client.SSHClient:
    if keyPath is not None:
        k = paramiko.RSAKey.from_private_key_file(keyPath)
    else:
        k = None
    sshc = paramiko.SSHClient()
    sshc.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sshc.connect(hostname, port, username, password, pkey=k)
    return sshc


def uploadFile(sourcePath: str, targetPath: str,
               sshc: paramiko.client.SSHClient) -> None:
    with scp.SCPClient(sshc.get_transport()) as scpc:
        scpc.put(sourcePath, targetPath)


def tdloadViaSSH(engine: sqlalchemy.engine.base.Engine,
                 sshc: paramiko.client.SSHClient,
                 tablename: str, targetFile: str, dbname: str = None,
                 jobname: str = "jobtmp", skipRowNum: int = 0,
                 verbose: bool = True) -> None:

    if dbname is None:
        dbname = engine.url.database
    # always use option: QuotedData = 'Optional'
    options = "--DCPQuotedData 'Optional'"
    if skipRowNum > 0:
        options += f" --SourceSkipRows {skipRowNum}"

    tdload_command = (f"tdload -f {targetFile} -t {dbname}.{tablename}"
                      + f" -h {engine.url.host} -u {engine.url.username}"
                      + f" -p {engine.url.password}"
                      + f" --TargetWorkingDatabase {dbname}"
                      + f" {options} {jobname}")

    # drop error log table if exists
    dropIfExists(tablename+"_ET", dbname, engine)
    dropIfExists(tablename+"_UV", dbname, engine)

    # execute command via ssh
    stdin, stdout, stderr = sshc.exec_command(tdload_command)
    for line in stdout:
        if verbose:
            print(line)
        else:
            if re.match(r".*(Total Rows|successfully).*", line):
                print(line)


def tdload_df(df: pd.DataFrame, engine: sqlalchemy.engine.base.Engine,
              tablename: str, ssh_ip: str, ssh_username: str,
              dbname: str = None, overwrite: bool = False,
              ssh_password: str = None, ssh_keypath: str = None,
              ssh_folder: str = None, saveIndex=False,
              indexList: list = None, isIndexUnique: bool = True,
              verbose: bool = True) -> None:

    # 1. save csv
    if dbname is None:
        dbname = engine.url.database
    # use time for avoiding overwrite existing file
    sourceName = f"tmp_{dbname}_{tablename}_{time.time():.0f}.csv"
    if saveIndex:
        df = df.reset_index()
    df.to_csv(sourceName, index=False)
    # by reading csv again, we can get same string format as csv
    df_tmp = pd.read_csv(sourceName)

    # 2. create table
    createTable(df_tmp, engine, tablename, dbname,
                overwrite, indexList, isIndexUnique)

    # 3. copy file with scp
    if ssh_folder is None:
        ssh_folder = "~"
    targetPath = "/" + sourceName

    sshc = connectSSH(ssh_ip, ssh_username, ssh_password, keyPath=ssh_keypath)
    uploadFile(sourceName, targetPath, sshc)

    # 4. load file with tdload
    tdloadViaSSH(engine=engine, sshc=sshc, tablename=tablename,
                 targetFile=targetPath, dbname=dbname,
                 skipRowNum=1, verbose=verbose)

    # 5. delete tmp files and close connection
    os.remove(sourceName)
    sftp = sshc.open_sftp()
    sftp.remove(targetPath)
    sshc.close()
