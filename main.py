import pandas as pd
import os
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

DATA_FOLDER = '/Users/ben/Desktop/cQuant/energy-analyst-data-exercise-public-master/historicalPriceData'
SAVE_ROOT = '/Users/ben/Desktop/cQuant/'

def task_1(data_folder):
    data_paths = sorted(glob.glob(os.path.join(data_folder, '*.csv')))
    main_df = None
    for path in data_paths:
        df = pd.read_csv(path)
        if main_df is None:
            main_df = df
        else:
            main_df = pd.concat([main_df, df])
    return main_df

def task_2(all_historical_df):
    historical_data_all_points = list(all_historical_df['SettlementPoint'].unique())
    all_settlementpoints = {}
    price_points = []
    for settlementpoint in historical_data_all_points:
        settlement_point_avg = {}
        settlementpoint_df = all_historical_df[all_historical_df['SettlementPoint'] == settlementpoint]
        curr_year = '2016'
        curr_month = '01'
        for ind, row in settlementpoint_df.iterrows():
            
            month = row['Date'].split('-')[1]
            year = row['Date'].split('-')[0]
            if curr_year == year:
                if curr_month == month:
                    price_points.append(row['Price'])
                    
                else:
                    price_point_avg = np.mean(np.array(price_points))
                    settlement_point_avg[(curr_year, curr_month)] = price_point_avg
                    curr_year = year
                    curr_month = month
                    price_points = []
            else:
                price_point_avg = np.mean(np.array(price_points))
                settlement_point_avg[(curr_year, curr_month)] = price_point_avg
                curr_year = year
                curr_month = month
                price_points = []
                
        all_settlementpoints[settlementpoint] = settlement_point_avg

    all_settlement_points = []
    all_years = []
    all_months = []
    all_prices = []
    for settlement_point, avg_prices in all_settlementpoints.items():
        months_years = list(avg_prices.keys())
        for month_year in months_years:
            all_settlement_points.append(settlement_point)
            all_years.append(month_year[0])
            all_months.append(month_year[1])
            all_prices.append(avg_prices[month_year])
    
    averagePriceByMonth = pd.DataFrame({
    'SettlementPoint':all_settlement_points,
    'Year':all_years,
    'Month':all_months,
    'AveragePrice':all_prices
    })
    return averagePriceByMonth
    

def task_3(averagePriceByMonth):
    averagePriceByMonth.to_csv('AveragePriceByMonth.csv')

def task_4(all_historical_df):
    task4_df = all_historical_df[all_historical_df['SettlementPoint'].str.contains("HB")]
    task4_df = task4_df.loc[~(task4_df['Price']<=0)]
    settlement_hb = task4_df['SettlementPoint'].unique()
    log_returns_full = {}
    log_returns_calc = lambda ri, rf: np.log(rf) - np.log(ri)
    for point in settlement_hb:
        sp_df = task4_df[task4_df['SettlementPoint'] == point]
        price_points_2016 = sp_df[sp_df['Date'].str.contains('2016')]['Price'].values
        
        log_returns_2016 = np.asarray([log_returns_calc(price_points_2016[i], rf) for i,rf in enumerate(price_points_2016[1:])])
        price_points_2017 = sp_df[sp_df['Date'].str.contains('2017')]['Price'].values
        log_returns_2017 = np.asarray([log_returns_calc(price_points_2017[i], rf) for i,rf in enumerate(price_points_2017[1:])])
        price_points_2018 = sp_df[sp_df['Date'].str.contains('2018')]['Price'].values
        log_returns_2018 = np.asarray([log_returns_calc(price_points_2018[i], rf) for i,rf in enumerate(price_points_2018[1:])])
        price_points_2019 = sp_df[sp_df['Date'].str.contains('2019')]['Price'].values
        log_returns_2019 = np.asarray([log_returns_calc(price_points_2019[i], rf) for i,rf in enumerate(price_points_2019[1:])])
        
        log_returns_full[point] = [
            np.std(log_returns_2016), 
            np.std(log_returns_2017), 
            np.std(log_returns_2018), 
            np.std(log_returns_2019)]

    
    return log_returns_full

def task_5(log_returns_full):
    all_sp_points = []
    all_years = []
    all_volatilities = []
    for sp_point, hourly in log_returns_full.items():
        for i, vol in enumerate(hourly):
            
            if np.isnan(vol) == False:
                all_volatilities.append(vol)
                all_years.append(i+2016)
                all_sp_points.append(sp_point)
    hourlyVolatitlityByYear = pd.DataFrame({
        'SettlementPoint':all_sp_points,
        'Year':all_years,
        'HourlyVolatility':all_volatilities
        })
    hourlyVolatitlityByYear.to_csv('HourlyVolatitlityByYear.csv')
    

    return hourlyVolatitlityByYear

def task_6(hourlyVolatitlityByYear):
    years = hourlyVolatitlityByYear['Year'].unique()
    max_rows = []
    for year in years:
        max_val = hourlyVolatitlityByYear.iloc[hourlyVolatitlityByYear[hourlyVolatitlityByYear['Year']==year]['HourlyVolatility'].idxmax()]
        max_rows.append(max_val)
    MaxVolatilityByYear = pd.DataFrame(max_rows)
    MaxVolatilityByYear.reset_index(inplace = True)
    MaxVolatilityByYear.drop(columns = ['index'], inplace = True)
    MaxVolatilityByYear.to_csv('MaxVolatilityByYear.csv')
    

def task_7(all_historical_df):
    pd.options.mode.chained_assignment = None
    settlement_points = all_historical_df['SettlementPoint'].unique()
    nan_dict = {}
    for i in range(1, 25):
        nan_dict[f'X{i}'] = np.nan
        
    for sp_val in settlement_points:

        points_all = all_historical_df[all_historical_df['SettlementPoint'] == sp_val]
        points_all[['Date','Hour']] = points_all['Date'].str.split(' ',expand=True)
        unique_dates = points_all['Date'].unique()
        spot_vals = []

        for date in unique_dates:
            curr_date_vals = {}
            curr_date_vals = nan_dict.copy()
            curr_date = points_all[points_all['Date'] == date]
            curr_date_vals['Variable'] = 'HB_BUSAVG'
            curr_date_vals['Date'] = date
            curr_date[['Hour','Min','Second']] = curr_date['Hour'].str.split(':', expand = True)  
            curr_date_hours = curr_date['Hour'].values
            hours = np.array(curr_date_hours, dtype = np.uint8)
            for hour in curr_date_hours:
                price = curr_date[curr_date['Hour']==hour]['Price'].values
                curr_date_vals[f'X{int(hour)+1}'] = price[0]
            spot_vals.append(curr_date_vals)
        
        spot_vals_df = pd.DataFrame(spot_vals)
        spot_vals_df_cols = list(spot_vals_df.columns)
        cols = spot_vals_df_cols[-2:] + spot_vals_df_cols[:-2] 
        spot_vals_df = spot_vals_df[cols].to_csv(os.path.join(args.save_root, 'formattedSpotHistory',f'spot_{sp_val}.csv'))

def line_plot(unique_items, items_df, settlementpoint_type = 'Hub'):
    new_date_vals = []
    for ind, row in items_df.iterrows():
        if int(row['Month']) < 10:
            new_date_vals.append(f"{row['Year']}-0{row['Month']}-01")
        else:
            new_date_vals.append(f"{row['Year']}-{row['Month']}-01")

    items_df['YearMonth'] = new_date_vals
    items_data = {}
    for unique in unique_items:
        prices = items_df[items_df['SettlementPoint'] == items_df]['AveragePrice'].values
        year_months = items_df[items_df['SettlementPoint'] == items_df]['YearMonth'].values
        items_data[unique] = [prices, year_months]

    plt.figure(figsize = (15,15))
    for unique in unique_items:
        plt.plot(items_data[unique][1], items_data[unique][0])
    plt.legend(unique_items)
    if settlementpoint_type == 'Hub':
        plt.title('Settlement Hub Average Price By Month')
        plt.savefig('SettlementHubAveragePriceByMonth.png')
    else:
        plt.title('Load Zone Average Price By Month')
        plt.savefig('LoadZoneAveragePriceByMonth.png')

    

def bonus_meanplots(averagePriceByMonth):
    hubs = averagePriceByMonth[averagePriceByMonth['SettlementPoint'].str.contains('HB_')]
    lzs = averagePriceByMonth[averagePriceByMonth['SettlementPoint'].str.contains('LZ_')]
    hubs_unique = hubs['SettlementPoint'].unique()
    lzs_unique = lzs['SettlementPoint'].unique()

    line_plot(hubs_unique, hubs, 'Hub')
    line_plot(lzs_unique, lzs, 'LZ')
    


def bonus_volatilityplots(hourlyVolatitlityByYear):
    unique_hubs = hourlyVolatitlityByYear['SettlementPoint'].unique()
    plt.figure(figsize = (10,5))
    for hub in unique_hubs:
        hub_data = hourlyVolatitlityByYear[hourlyVolatitlityByYear['SettlementPoint'] == hub]
        years = hub_data['Year'].values
        vol = hub_data['HourlyVolatility'].values
        plt.plot(years, vol)
        

    plt.legend(unique_hubs)
    plt.title('Volatility Plot of Settlement Hubs by Year')
    plt.savefig('VolatilityPlotSettlementHubsYear.png')


def main(args):
    try:
        os.makedirs(args.save_root)
    except OSError:
        pass

    
    # task 1
    all_historical_df = task_1(args.data_folder)
    # task 2
    
    
    averagePriceByMonth = task_2(all_historical_df)
    # task 3
    
    task_3(averagePriceByMonth)
    # task 4
    log_returns_df = task_4(all_historical_df)
    # task 5
    hourlyVolatitlityByYear = task_5(log_returns_df)
    # task 6
    task_6(hourlyVolatitlityByYear)
    # task 7
    
    task_7(all_historical_df)
    
    # bonus 1
    bonus_meanplots(averagePriceByMonth)

    # bonus 2
    bonus_volatilityplots(hourlyVolatitlityByYear)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder',
        default = DATA_FOLDER,
        type = str,
        help = 'Put folder to all csv files here'
    )
    parser.add_argument(
        '--save_root',
        default = SAVE_ROOT,
        type = str
    )
    args = parser.parse_args()
    main(args)

