ps -ef | grep python | awk '{print $2}' | xargs sudo kill -9
ps -ef | grep zzz | awk '{print $2}' | xargs sudo kill -9
ps -ef | grep rosout | awk '{print $2}' | xargs sudo kill -9
