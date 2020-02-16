from sqlalchemy import create_engine, Integer, Float, String
import numpy as np
import pandas as pd
import paramiko, sqlalchemy, scp, time, os, re, pathlib, subprocess


def tableIsExist(tablename: str, dbname: str,
                 engine: sqlalchemy.engine.base.Engine) -> bool:
    """
    ===example===
    from sqlalchemy import create_engine
    engine = create_engine("teradata://dbc:dbc@hostip:1025")
    tableIsExist("iris", "db_data", engine)
    >>> True
    """
    table_list = pd.read_sql_query(
        """select * from dbc.TablesV
        where TableName = '{tablename}'
        and DatabaseName = '{dbname}'""".format(
            dbname=dbname, tablename=tablename), engine)
    return True if len(table_list) > 0 else False


def dropIfExists(tablename: str, dbname: str,
                 engine: sqlalchemy.engine.base.Engine) -> None:
    """
    ===example===
    from sqlalchemy import create_engine
    engine = create_engine("teradata://dbc:dbc@hostip:1025")
    dropIfExists("iris", "db_data", engine)
    droped table db_data.iris
    >>> None
    """
    if tableIsExist(tablename, dbname, engine):
        engine.execute(f"drop table {dbname}.{tablename}")
        print(f"droped table {dbname}.{tablename}")


def dtypeParser(series: pd.Series) -> str:
    """
    function to change pandas data type to Teradata data type
    ===example===
    dtypeParser(df["petal_length"])
    >>> "float"
    """
    if series.dtype.name.startswith("int"):
        if series.max() < np.iinfo(np.int32).max and \
                series.min() > np.iinfo(np.int32).min:
            parsed_dtype = "integer"
        elif series.max() < np.iinfo(np.int64).max and \
                series.min() > np.iinfo(np.int64).min:
            parsed_dtype = "bigint"
        else:
            max_length = max(1, max(series.astype(str).str.len())
                             )  # at least 1 length
            parsed_dtype = f"varchar({max_length})"
    elif series.dtype.name.startswith("float64"):
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
                ifExists: str = "error", indexList: list = None,
                isIndexUnique: bool = True, dtype: dict = {}
                ) -> None:
    """
    ===example===
    createTable(df_iris, engine, "iris", "db_data",
                ifExists="replace")
    >>> None
    """
    # make query string
    if dbname is None:
        dbname = engine.url.database
    new_names = renameColumns(df.columns.to_series())
    col_dtype = [dtypeParser(col) if name not in dtype else dtype[name]
                 for name, col in df.iteritems()]

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
    if ifExists == "replace":
        dropIfExists(tablename, dbname, engine)
    elif ifExists == "error":
        if tableIsExist(tablename, dbname, engine):
            raise ValueError(f"{tablename}.{dbname} is already exists.")
    elif ifExists == "insert":
        if tableIsExist(tablename, dbname, engine):
            return

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


def archiveFile(sourcePath: str, archivePath: str = None, method: str = "gz",
                verbose: bool = True) -> str:
    if archivePath is None:
        archivePath = str(sourcePath) + "." + method

    if method == "7z":
        command = f"7z a {archivePath} {sourcePath}"
    elif method in ["gz", "xz", "bz2"]:
        sourceName = pathlib.Path(sourcePath).name
        sourceFolder = pathlib.Path(sourcePath).parent
        d_format_option = {"gz": "z", "xz": "J", "bz2": "j"}
        command = (f"tar -C {sourceFolder} -{d_format_option[method]}cvf"
                   + f" {archivePath} {sourceName}")
    else:
        raise ValueError(f"method only supports ['7z', 'gz', 'xz', 'bz2']")

    with verbosity_context(f"Archiving {sourcePath} to {archivePath}",
                           verbose):
        if verbose:
            print(subprocess.check_output(command,
                                          universal_newlines=True, shell=True))
        else:
            subprocess.check_output(
                command, universal_newlines=True, shell=True)
    return archivePath


def unarchiveSSH(archivePath: pathlib.PosixPath,
                 sshc: paramiko.client.SSHClient,
                 unarchiveFolder: pathlib.PosixPath = None,
                 method: str = "gz", verbose: bool = True
                 ) -> None:
    if unarchiveFolder is None:
        unarchiveFolder = archivePath.parent

    if method == "7z":
        command = f"7z e {archivePath} -o{unarchiveFolder}"
    elif method in ["gz", "xz", "bz2"]:
        d_format_option = {"gz": "z", "xz": "J", "bz2": "j"}
        command = (f"tar -xv{d_format_option[method]}f"
                   + f"{archivePath} -C {unarchiveFolder}")
    else:
        raise ValueError(f"method only supports ['7z', 'gz', 'xz', 'bz2']")

    with verbosity_context(f"Unarchiving {archivePath}", verbose):
        stdin, stdout, stderr = sshc.exec_command(command)
        if verbose:
            for line in stdout:
                print(line)


def uploadFile(sourcePath: str, targetPath: str,
               sshc: paramiko.client.SSHClient,
               compress_method: str = None,
               verbose: bool = True) -> pathlib.Path:
    def show_progress(filename, size, sent):
        print(f"Uploading {filename} progress: " +
              f"{float(sent)/float(size)*100:.2f}%", end="\r")
    progress = show_progress if verbose else None

    try:
        if compress_method:
            fileName = pathlib.Path(sourcePath).name
            # change targetPath for uploading to
            # targetPath's directory / sourcePath's name + ext.
            targetPath = pathlib.Path(
                str(pathlib.Path(targetPath).parent / fileName) + "."
                + compress_method)
            sourcePath = archiveFile(sourcePath, verbose=verbose,
                                     method=compress_method)
            isArchived = True

        with scp.SCPClient(sshc.get_transport(), progress=progress) as scpc:
            # in case Path is PosixPath, casting them to str
            scpc.put(str(sourcePath), str(targetPath))
            print("\n")  # nextline

        if compress_method:
            unarchiveSSH(targetPath, sshc, method=compress_method,
                         verbose=verbose)
            isUnarchived = True
            # change targetPath to uploaded raw file
            uploadedPath = str(pathlib.Path(targetPath).parent/fileName)
    finally:  # delete archive files
        if 'isArchived' in locals():
            with verbosity_context(f"Deleting archive {sourcePath}",
                                   verbose):
                os.remove(sourcePath)
        if 'isUnarchived' in locals():
            sftp = sshc.open_sftp()
            with verbosity_context(f"Deleting archive {targetPath} via SCP",
                                   verbose):
                sftp.remove(str(targetPath))

    return uploadedPath if compress_method else targetPath


def tdloadViaSSH(engine: sqlalchemy.engine.base.Engine,
                 sshc: paramiko.client.SSHClient,
                 tablename: str, targetPath: str,
                 dbname: str = None,
                 jobname: str = "jobtmp", skipRowNum: int = 0,
                 verbose: bool = True) -> None:

    targetPath = pathlib.Path(targetPath)
    if dbname is None:
        dbname = engine.url.database
    # always use option: QuotedData = 'Optional'
    options = "--DCPQuotedData 'Optional'"
    if skipRowNum > 0:
        options += f" --SourceSkipRows {skipRowNum}"

    tdload_command = (f"tdload -f {targetPath} -t {dbname}.{tablename}"
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


class verbosity_context:
    def __init__(self, processName: str, verbose: bool = True):
        self.verbose = verbose
        self.processName = processName

    def __enter__(self):
        self.startTime = time.time()
        if self.verbose:
            print(f"start {self.processName}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.verbose:
            elapsedTime = time.time() - self.startTime
            print(f"    ...finished. : "
                  + f"elapsed time = {elapsedTime:.2f}.sec")


def tdload_df(df: pd.DataFrame, engine: sqlalchemy.engine.base.Engine,
              tablename: str, ssh_ip: str, ssh_username: str,
              dbname: str = None, ifExists: str = "error",
              compress: str = None, dtype: dict = {},
              ssh_password: str = None, ssh_keypath: str = None,
              ssh_folder: str = None,
              dump_folder: str = None,
              saveIndex=False, indexList: list = None,
              isIndexUnique: bool = True, verbose: bool = True) -> None:

    try:
        # 1. save csv
        if dbname is None:
            dbname = engine.url.database
        # use time for avoiding overwrite existing file
        sourceName = f"tmp_{dbname}_{tablename}_{time.time():.0f}.csv"
        if dump_folder is None:
            dump_folder = pathlib.Path().cwd()
        sourcePath = pathlib.Path(dump_folder)/sourceName

        if saveIndex:
            df = df.reset_index()
        with verbosity_context(f"dumping DF to {sourcePath}", verbose):
            csvMade = df.to_csv(sourcePath, index=False)
        # by reading csv again, we can get same string format as csv
        with verbosity_context(f"re-reading {sourcePath}", verbose):
            df = pd.read_csv(sourcePath)

        # 2. create table
        createTable(df, engine, tablename, dbname,
                    ifExists, indexList, isIndexUnique, dtype)

        # 3. copy file with scp
        with verbosity_context(f"connecting ssh", verbose):
            sshc = connectSSH(ssh_ip, ssh_username,
                              ssh_password, keyPath=ssh_keypath)
            isSSHConnected = True
        if ssh_folder is None:
            _, tmp_out, _ = sshc.exec_command('pwd')
            ssh_folder = tmp_out.readline().strip()
        targetPath = pathlib.Path(ssh_folder)/sourceName
        with verbosity_context(f"Uploading File {sourcePath} to {targetPath}",
                               verbose):
            uploadedPath = uploadFile(sourcePath, targetPath, sshc, compress,
                                      verbose)

        # 4. load file with tdload
        with verbosity_context(f"Loading File {uploadedPath} to DB", verbose):
            tdloadViaSSH(engine=engine, sshc=sshc, tablename=tablename,
                         targetPath=uploadedPath, dbname=dbname,
                         skipRowNum=1, verbose=verbose)

    finally:
        # 5. delete tmp files and close connection
        if 'csvMade' in locals():
            with verbosity_context(f"Deleting dumped file {sourcePath}",
                                   verbose):
                os.remove(sourcePath)
        if 'uploadedPath' in locals():
            sftp = sshc.open_sftp()
            with verbosity_context(f"Deleting {uploadedPath} via SCP",
                                   verbose):
                sftp.remove(str(uploadedPath))
        if 'isSSHConnected' in locals():
            with verbosity_context("Closing SSH", verbose):
                sshc.close()
