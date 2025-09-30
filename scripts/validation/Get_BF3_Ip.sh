#!/usr/bin/expect -f


puts -nonewline "Enter the device name(rshim0,...): "
flush stdout
set timeout 30

expect_user -re "(.*)\n" {
    set input $expect_out(1,string)
}


# set input [lindex $argv 0]
set device "/dev/$input/console"
set baud_rate "115200"
set username "root"
set password "Supermicro12345"
set command "ip a | grep -i 'oob_net'"
# set command "cat /opt/mellanox/doca/applications/VERSION" 

# Start screen
spawn screen $device -b $baud_rate

# Expect prompt for username
expect "login:"
puts "login..."
send "$username\r"

# Expect prompt for password
expect "Password:"
send "$password\r"

# Wait for command prompt
expect "$ "

# Send command
puts "getting oob_IP....."
send "$command\r"

# Wait for command output
expect "$ "

# Exit screen
send "\x01:" ; # Sends Ctrl-A and :
send "quit\r"

# Wait for screen to exit
expect eof

