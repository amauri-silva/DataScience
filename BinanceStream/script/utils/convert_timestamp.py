from datetime import datetime

def get_date_time_from_timestamp(timestamp):
    """
    Get date and time from timestamp 
    And return three values
    full_date_time --> Ex: 2024-03-09 18:34:43.805000
    only_date --> Ex: 2024-03-09
    only_time --> Ex: 18:34:43.805000
    """
    t_tamp_caculated = timestamp / 1000
    print("beforeeeeeeeeeeeeeeeee")
    full_date_time = datetime.fromtimestamp(t_tamp_caculated)
    print("afteeeeeeeeeeerrrrrrrrr")
    only_date, only_time = full_date_time.date(), full_date_time.time()
    return only_date, only_time
