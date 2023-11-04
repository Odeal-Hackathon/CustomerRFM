import pandas as pd

class RFM:
    def __init__(self, filepath):
        """
        Initialize the RFM class with the given data file path.

        Parameters
        ----------
        filepath : str
            Path to the data file in Parquet format.
        """
        self.filepath = filepath
        self.df = self._load_data()
        self.latest_date = self.df['IslemTarih'].max()
        self.rfm = self._calculate_rfm()

    def _load_data(self):
        """
        Load data from the Parquet file and convert the transaction date column to datetime.

        Returns
        -------
        DataFrame
            The data loaded from the Parquet file with the 'IslemTarih' column as datetime.
        """
        df = pd.read_parquet(self.filepath, engine='fastparquet')
        df['IslemTarih'] = pd.to_datetime(df['IslemTarih'])
        return df

    def _dynamic_qcut(self, column, labels):
        """
        Dynamically bin values into quantiles with customized labels, dropping duplicates.

        Parameters
        ----------
        column : Series
            Pandas Series object to be binned.
        labels : list
            Labels for the resulting bins.

        Returns
        -------
        Series
            Binned values of the input series with custom labels.
        """
        _, bins = pd.qcut(column, q=4, retbins=True, duplicates='drop')
        n = len(bins) - 1
        return pd.cut(column, bins=bins, labels=labels[-n:], duplicates='drop', include_lowest=True)

    def _calculate_rfm(self):
        """
        Calculate the RFM (Recency, Frequency, Monetary) metrics for each customer.

        Returns
        -------
        DataFrame
            RFM metrics calculated for each customer ID.
        """
        rfm = self.df.groupby('id').agg({
            'IslemTarih': lambda date: (self.latest_date - date.max()).days,
            'IslemID': 'count',
            'IslemTutar': 'sum'
        }).rename(columns={'IslemTarih': 'Recency', 'IslemID': 'Frequency', 'IslemTutar': 'Monetary'})
        
        rfm['R'] = self._dynamic_qcut(rfm['Recency'], labels=[4, 3, 2, 1])
        rfm['F'] = self._dynamic_qcut(rfm['Frequency'], labels=[1, 2, 3, 4])
        rfm['M'] = self._dynamic_qcut(rfm['Monetary'], labels=[1, 2, 3, 4])
        
        rfm['RFM_Segment'] = rfm.apply(lambda x: f"{x['R']}{x['F']}{x['M']}", axis=1)
        rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
        
        return rfm

    def _get_rfm_status(self, score, avg_score, std_score):
        """
        Determine the RFM status based on score thresholds.

        Parameters
        ----------
        score : int or float
            The RFM score of a particular customer.
        avg_score : float
            The average RFM score of all customers.
        std_score : float
            The standard deviation of RFM scores among all customers.

        Returns
        -------
        str
            The RFM status of the customer.
        """
        if score >= (avg_score + std_score):
            return "Platinum"
        elif (avg_score + std_score) > score >= avg_score:
            return "Altın"
        elif avg_score > score >= (avg_score - std_score):
            return "Gümüş"
        else:
            return "Bronz"

    def _get_rfm_description(self, status):
        """
        Retrieve a description for a given RFM status.

        Parameters
        ----------
        status : str
            The RFM status.

        Returns
        -------
        str
            A description of the RFM status.
        """
        descriptions = {
            "Platinum": "Yüksek değerli müşteri. Ortalamanın çok üzerinde sıklık ve para değeri.",
            "Altın": "Orta değerli müşteri. Ortalamanın üzerinde sıklık ve para değeri.",
            "Gümüş": "Düşük-orta değerli müşteri. Ortalamanın altında sıklık ve para değeri, ancak en düşük değil.",
            "Bronz": "Düşük değerli müşteri. Ortalamanın çok altında sıklık ve para değeri."
        }
        return descriptions.get(status, "Tanımsız")

    def calculate_customer_value(self):
        """
        Compute the customer value scores and assign RFM status and description.
        """
        avg_score = self.rfm['RFM_Score'].mean()
        std_score = self.rfm['RFM_Score'].std()

        self.rfm['Status'] = self.rfm['RFM_Score'].apply(self._get_rfm_status, args=(avg_score, std_score))
        self.rfm['Description'] = self.rfm['Status'].apply(self._get_rfm_description)

    def save_results(self, output_file):
        """
        Save the RFM results to a Parquet file.

        Parameters
        ----------
        output_file : str
            The path where the RFM results will be saved.
        """
        self.rfm.reset_index().to_parquet(output_file)


if __name__ == "__main__":
    calculator = RFM('data/odeal_hackathon.parquet')
    calculator.calculate_customer_value()
    calculator.save_results("data/rfm.parquet")
