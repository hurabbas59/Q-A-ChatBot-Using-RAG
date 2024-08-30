from bot import run
#test bot on terminal
while True:

    query = input("Enter Query: ")
    response = run(query)
    print(response)

    exit_cmd = int(input("Exit 0"))
    if exit_cmd==0:
        break