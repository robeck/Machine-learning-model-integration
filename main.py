# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def test_exception():
    h =3
    try:
        fh = open('testfile','r')
        fh.write('it is a test file')
        h = 0
    except:
        print('error')
        h = 1

    print('code should be running')
    print(h)




    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_exception()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
