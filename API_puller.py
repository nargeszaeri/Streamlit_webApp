import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import concurrent.futures
from tqdm import tqdm

def API_puller(trend_log_list, API_key, date_range, resample=None, max_workers=None):
    """Retrieves data from Coppertree Analytics Kaizen API and organizes it into a single dataframe

    This function utilizes multithreading for to increase speed of API processing

    Args:
        trend_log_list (Pandas Dataframe):
            a two column pandas dataframe with the trend log controller number in the in the first column
            and the name of the trend log in the second column
        
        API_key (str):
            Your api key, which can be accessed through you're Kaizen account
            
        date_range (list, Format: ['YYYY-MM-DD', 'YYYY-MM-DD']):
            a list of two date strings indicating start date and end date.
            Note: The date range is non inclusive, so the "end date" is not included in the API call
        
        resample (int, optional): Defaults to None.
            Resample dataframe in minutes. For example to resample every 1 hour, enter resample=60. Fill method
            is based on previous within the resample time frame. If there is no samples, NaN is returned
            If none is received, no resampling will occur (warning: this may result in large outputs if
            event based sensors are included in query). 
                
        max_workers (int, optional): Defaults to None.
            The number of threads that will be used to perform API calls. Use None in most cases. Lower numbers
            may reduce errors by reducing frequency of calls to the API. Some trial and error is required here.

    Returns:
        Dataframe:
            Organized dataframe of the requested sensor inputs
    """

    trend_log_dict = trend_log_list.to_records(index=False)

    def save_api_data(trend_log):
        """ Employs the getData function and appends the results to a list.
        This is used for the thread pool executor to allow for multi-threading
        """
        df = getData(trend_log[0], trend_log[1], date_range[0], date_range[1], API_key, resample)
        dfs.append(df)

    # Perform API calls using multi-threading
    dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        progress_bar = tqdm(executor.map(save_api_data, trend_log_dict), total=len(trend_log_dict))
        progress_bar.set_description('API Download')
        list(progress_bar)

    # Combine each log into a main dataframe
    pbar = tqdm(dfs)
    pbar.set_description('Organizing')
    df_concat = pd.DataFrame()
    for df in pbar:
        df_concat = pd.concat([df_concat, df], axis=1, join="outer")


    # Check that all trends were downloaded
    logs = trend_log_list.iloc[:, 1].tolist()
    if not all(item in df_concat.columns for item in logs):
        raise Exception("No all logs were downloaded. In some cases, reducing the max_workers may help with this issue")
    
    # Reorder columns based on input order
    df_concat = df_concat[logs]

    return df_concat


def getData(trend_log_ID, trend_log_name_ID, start, end, API_key, sample):
    """Uses Kaizen's public API to get trend log data and returns it as a pandas dataframe

    For handling empty rows, an empty dataframe is returned
    """

    url = 'https://kaizen.coppertreeanalytics.com/public_api/api/get_tl_data_start_end?&' \
          'api_key={}&tl={}&start={}T00:00:00&end={}T00:00:00&data=raw'.format(API_key, trend_log_ID, start, end)
    retry_strategy = Retry(total=5, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    results = http.get(url)
    check_response(results)

    # Retrieve results and convert to pandas dataframe
    # If statement is used to handle cases where api call retrieves and empty list
    if results.text == '[]':
        results = pd.DataFrame(index=pd.to_datetime([]), columns=[trend_log_name_ID])
        results.index.name = 'ts'
    else:
        results = pd.read_json(results.text)
        results = results.rename(columns={'v': trend_log_name_ID})
        results['ts'] = pd.to_datetime(results['ts'])
        results = results.set_index('ts')
        if isinstance(sample, int):
            results = results.resample(str(sample) + 'min').first()
    return results


def check_response(r):
    """Checks to ensure the expected response is received

    The accepted response from the API from the API is response [200] this
    function outputs raises an error if any other response is retrieved.
    """
    if r.status_code == 200:
        return None
    else:
        raise ImportError(f'Received: [<Response [{r.status_code}]>], Expected: [<Response [200]>]')


# You can use this if you have an CSV sensor list
# if __name__ == '__main__':
#     trend_log_list = pd.read_csv('Your_Sensor_list.csv', header=None)
#     df = API_puller(
#         trend_log_list=trend_log_list,
#         API_key='Get from account login or from Connor Brackley (connor.brackley@mail.concordia.ca',
#         date_range=['2015-12-31', '2022-01-01'],
#         resample=15
#     )
#     print(df)
