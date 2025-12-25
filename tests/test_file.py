from ms_decarrier import decarry_file

def test_file():
    decarry_file(
        'test.mgf',
        'decarry_test.mgf'
    )