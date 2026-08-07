# restart-demo.ps1
# Use this any time a server has gone down mid-demo (e.g. laptop slept,
# a tunnel dropped, "Agent disconnected" shows up). It does exactly what
# start-demo.ps1 does -- kill anything still running, start everything
# fresh, print the new link -- kept as a separate file so "start" vs
# "something broke, fix it" stay easy to tell apart from the file name.

& "$PSScriptRoot\start-demo.ps1"
