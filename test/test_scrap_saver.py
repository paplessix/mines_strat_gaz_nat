from scrap.scrap_saver import * 
### filename constructor

def test_price_type():
    try : 
        data_updater('.','foo')
    except KeyError as e :
        assert True




