


def check_prime(n):

    if n<=1:
        return False
    
    for i in range(2, n):

        if n%i == 0:
            return False
        
    return True




if __name__ == "__main__":
    x = check_prime(7)
    print(x)
