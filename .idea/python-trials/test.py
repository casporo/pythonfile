import comparison
import arrays as arr

try:
    i = ''
    while i != 'N':
        select = int(input("Key in: "))
        if select == 1 :
            a = int(input("Enter value for A:"))
            b = int(input("Enter value for B: "))
            comparison.compareTwoInput(a,b)
        elif select == 2:
            #z = arrays.person1["age"]
            z = arr.thislist[1]
            print(z)

        i = input("Do you wish to continue Y/N ? ").upper()
except:print("Syntax error")





