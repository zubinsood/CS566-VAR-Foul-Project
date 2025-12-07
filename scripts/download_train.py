from SoccerNet.Downloader import SoccerNetDownloader as SNdl

data_dir = "/path/to/SoccerNetData"

mySNdl = SNdl(LocalDirectory=data_dir)

password = "s0cc3rn3t"

mySNdl.downloadDataTask(
    task="mvfouls", 
    split=["train"],
    password=password
)