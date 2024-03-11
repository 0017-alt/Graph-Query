import re
import datetime
import numpy as np
from neo4j import GraphDatabase, Driver
import pandas as pd
from collections import OrderedDict
import spacy
import openai
import json

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI API
openai.api_key = "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"

def ask_gpt(insert_prompt):
  model_choice = "gpt-4"
  try:
    response = openai.ChatCompletion.create(
      model=model_choice,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": insert_prompt}
      ]
    )
    return response['choices'][0]['message']['content'].strip()
  except Exception as e:
    return f"An error occurred: {e}"


# Find the most related prompt
def find_most_related_prompt(text):
  gpt_input = """
    You follow the program below;

    Thare are template prompts as follws;
    PROMPT = [
      # 0
      "for setcor ENERG order according to totalAssets based on all financial statements",
      # 1
      "for sector ENERG order according to totalAssets based on latest financial statements",
      # 2
      "eps is 1 or more",
      "eps is under 1",
      # 3
      "safe stock order",
      # 4
      "big change in stock value on 2023/10/11",
      # 5
      "Arrange the stocks in ascending order based on the average stock price on 2023/10/11. with showing open, close",
      # 6
      "Arrange the stocks in sector ENERG in ascending order based on the average stock price on 2023/10/11 with showing open, close",
      # 7
      "Arrange the stocks in ascending order based on total assets in 2023",
      # 8
      "get average, open, close price on 2023/10/11",
      # 9
      "Arrange the stocks in ascending order based on total assets in quarter 3 of 2023",
    ]

    when the input is given, you return the number which is the most similar among the template prompts.
    Also, you should return the information of sector, feature of stock value(used to order data), list of stock value featue(not used to order data) date, year, feature of financial statement, condition, and quarter.
    The format id as follows;
    {"prompt": 1, "sector": "ENERG", "stock_value_value": "average", "stock_value_value_list": ["low", "high"], "date": "2023/10/11", "year": "2023", "financial_statement_value": "totalAssets", "condition": "epsQuarter > 1", "quarter": 3}
    If there is no proper value, the content shold be "".

    The value of stock_value_value is one of the following;
    STOCK_VALUE_VALUE = ["average", "close", "high", "low", "open", "prior", "totalVolume"]

    The value of financial_statement_value is one of the following;
    FINANCIAL_STATEMENT_VALUE = [
    "roa",
    "roe",
    "de",
    "financialStatementtype",
    "dateAsOf",
    "accountPeriod",
    "totalAssets",
    "totalAssetTurnover",
    "totalLiabilities",
    "paidupShareCapital",
    "shareholderEquity",
    "totalEquity",
    "totalRevenueQuarter",
    "totalRevenueAccum",
    "ebitQuarter",
    "ebitAccum",
    "netProfitQuarter",
    "netProfitAccum",
    "epsQuarter",
    "epsAccum",
    "operatingCacheFlow",
    "investingCashFlow",
    "financingCashFlow",
    "netProfitMarginQuarter",
    "netProfitMarginAccum",
    "fixedAssetTurnover"
    ]
    There are some values which have *Quarter and *Accum. If there is no mention in the input, select '*Quarter'.

    The value of financial_statement_value is one of the following;
    SECTOR = ["ICT", "TRANS", "PROP", "ENERG", "BANK", "HELTH", "FOOD", "TOURISM", "COMM", "ETRON", "PETRO", "FIN", "CONMAT", "PKG", "INSUR"]

    Conditions consist of stock_value_value and equal or inequality signs.

    Date is given in the format of yyyy/mm/dd or yyyy-mm-dd, and if the input has the format of yyyy-q, q shold show quarter.

    Now, you recieve '
    """ + \
   text + "'. Please return only the output following the format."
  gpt_output = ask_gpt(gpt_input)
  return gpt_output

def create_cypher_code(prompt_id, sector, stock_value_value, stock_value_value_list, date, year, financial_statement_value, condition, quarter):
  output_command = ""
  if prompt_id == 0:
    if sector == "" or financial_statement_value == "":
      return ""
    output_command = "MATCH (a:StandardIndustrialClassification {sectorCode:'" + \
      sector + \
      "'})<-[:BELONGING_SECTOR]-(b:Company)-[:REPORT]->(c:FinancialStatement)\nWITH c.year AS reportYear, c.quarter AS reportQuarter, b, c." + \
      financial_statement_value + \
      " AS " + \
      financial_statement_value + \
      "\nMATCH (b)-[:STOCK]->(d:Stock)\n" + \
      "WITH reportYear, reportQuarter, d.symbol AS symbol, " + financial_statement_value + \
      "\nRETURN symbol, reportYear, reportQuarter, " + \
      financial_statement_value + \
      "\nORDER BY " + financial_statement_value + " DESC"
  elif prompt_id == 1:
    if sector == "" or financial_statement_value == "":
      return ""
    output_command = "MATCH (c:FinancialStatement)\n" + \
      "WITH 10 * c.year + c.quarter AS yearQuarterSum\n" + \
      "WITH yearQuarterSum, COLLECT(yearQuarterSum) AS maxSums\n" + \
      "WITH MAX(maxSums)[0] AS maxSum\n" + \
      "MATCH (:StandardIndustrialClassification {sectorCode:'" + sector + "'})<-[:BELONGING_SECTOR]-(b:Company)-[:REPORT]->(c:FinancialStatement)\n" + \
      "WITH b, c." + financial_statement_value + " AS " + financial_statement_value + ", 10 * c.year + c.quarter AS yearQuarterSum, maxSum\n" + \
      "MATCH (b)-[:STOCK]->(d:Stock)\n" + \
      "WITH " + financial_statement_value + ", d.symbol AS symbol, yearQuarterSum, maxSum" + \
      "\nWHERE yearQuarterSum = maxSum\n" + \
      "RETURN symbol, " + financial_statement_value + \
      "\nORDER BY " + financial_statement_value + " DESC"
  elif prompt_id == 2:
    if condition == "":
      return ""
    output_command = "MATCH (c:FinancialStatement)\n" + \
      "WITH 10 * c.year + c.quarter AS yearQuarterSum\n" + \
      "WITH COLLECT(yearQuarterSum) AS maxSums\n" + \
      "WITH MAX(maxSums)[0] AS maxSum\n" + \
      "MATCH (a:Stock)<-[:STOCK]-(:Company)-[:REPORT]->(b:FinancialStatement)\n" + \
      "WHERE b." + condition + " AND 10 * b.year + b.quarter = maxSum\n" + \
      "WITH a.symbol AS symbol\n" + \
      "RETURN symbol"
  elif prompt_id == 3:
    output_command = "MATCH (a:FinancialStatement)\n" + \
      "WITH 10 * a.year + a.quarter AS yearQuarterSum\n" + \
      "WITH COLLECT(yearQuarterSum) AS maxSums\n" + \
      "WITH MAX(maxSums)[0] AS maxSum\n" + \
      "MATCH (b:Stock)<-[:STOCK]-(:Company)-[:REPORT]->(c:FinancialStatement)\n" + \
      "WITH b.symbol AS symbol, c.totalAssets AS totalAssets, c.totalLiabilities AS totalLiabilities, c.financingCashFlow AS financingCashFlow, c.netProfitQuarter AS netProfitQuarter, 10 * c.year + c.quarter AS yearQuarterSum, maxSum\n" + \
      "WITH symbol, totalLiabilities/totalAssets AS equityRatio, financingCashFlow, netProfitQuarter, yearQuarterSum, maxSum\n" + \
      "WHERE yearQuarterSum = maxSum AND equityRatio <= 0.5 AND financingCashFlow >= 0\n" + \
      "RETURN symbol, netProfitQuarter, equityRatio, financingCashFlow\n" + \
      "ORDER BY netProfitQuarter DESC\n" + \
      "LIMIT 5"
  elif prompt_id == 4:
    if date == "":
      return ""
    today = datetime.datetime.strptime(date, '%Y/%m/%d')
    yesterday = today + datetime.timedelta(days=-1)
    yesterday = "{:%Y/%m/%d}".format(yesterday).replace("/0", "/")
    output_command = 'MATCH (todayPrice:StockValue {date: "' + date + '"})<-[:STOCK_VALUE]-(stock:Stock)\n' + \
      'MATCH (stock)-[:STOCK_VALUE]->(yesterdayPrice:StockValue {date: "' + yesterday + '"})\n' + \
      'WITH stock.symbol AS symbol, todayPrice, yesterdayPrice\n' + \
      'RETURN symbol, todayPrice.average - yesterdayPrice.average AS priceDifference\n' + \
      'ORDER BY priceDifference DESC'
  elif prompt_id == 5:
    if date == "" or stock_value_value == "" or stock_value_value_list == "":
      return ""
    output_command = 'MATCH (a:Stock)-[r:STOCK_VALUE {date: "' + date + '"}]->(b:StockValue)\n' + \
      "WITH a.symbol AS symbol, b." + stock_value_value + " AS " + stock_value_value + ", "
    for s in stock_value_value_list:
      output_command += "b." + s + " AS " + s + ", "
    output_command = output_command[:-2]
    output_command += "\nRETURN symbol, " + stock_value_value + ", "
    for s in stock_value_value_list:
      output_command += s + ", "
    output_command = output_command[:-2] + "\nORDER BY " + stock_value_value + " DESC"
  elif prompt_id == 6:
    if sector == "" or date == "" or stock_value_value == "" or stock_value_value_list == "":
      return ""
    output_command = "MATCH (:StandardIndustrialClassification {sectorCode: '" + sector + "'})<-[:BELONGING_SECTOR]-(:Company)-[:STOCK]->(c:Stock)-[:STOCK_VALUE {date: '" + date + "'}]->(d:StockValue)\n" + \
      "WITH c.symbol AS symbol, d." + stock_value_value + " AS " + stock_value_value + ", "
    for s in stock_value_value_list:
      output_command += "d." + s + " AS " + s + ", "
    output_command = output_command[:-2] + "\nRETURN symbol, " + stock_value_value + ", "
    for s in stock_value_value_list:
      output_command += s + ", "
    output_command = output_command[:-2] + "\nORDER BY " + stock_value_value + " DESC"
  elif prompt_id == 7:
    if year == "" or financial_statement_value == "":
      return ""
    output_command = "MATCH (a:Stock)<-[:STOCK]-(:Company)-[:REPORT]->(b:FinancialStatement)\n" + \
      "WHERE " + "b.year = " + str(year) + "\n" + \
      "WITH a.symbol AS symbol, b.year AS year, b.quarter AS quarter, b." + financial_statement_value + " AS " + financial_statement_value + "\n" + \
      "RETURN symbol, year, quarter, " + financial_statement_value + \
      "\nORDER BY " + financial_statement_value + " DESC"
  elif prompt_id == 8:
    if date == "ERROR" or stock_value_value_list == "ERROR":
      return ""
    output_command = "MATCH (c:Stock)-[:STOCK_VALUE {date: '" + date + "'}]->(d:StockValue)\n" + \
        "WITH c.symbol AS symbol, "
    for s in stock_value_value_list:
      output_command += "d." + s + " AS " + s + ", "
    output_command = output_command[:-2]
    output_command += "\nRETURN symbol, "
    for s in stock_value_value_list:
      output_command += s + ", "
    output_command = output_command[:-2]
  elif prompt_id == 9:
    if year == "" or financial_statement_value == "":
      return ""
    output_command = "MATCH (a:Stock)<-[:STOCK]-(:Company)-[:REPORT]->(b:FinancialStatement)\n" + \
      "WHERE " + "b.year = " + str(year) + " AND b.quarter = " + str(quarter) + "\n" + \
      "WITH a.symbol AS symbol, b." + financial_statement_value + " AS " + financial_statement_value + "\n" + \
      "RETURN symbol, " + financial_statement_value + \
      "\nORDER BY " + financial_statement_value + " DESC"

  return output_command

def main(input_command):
  if input_command == "":
    return "error", ""

  gpt_output = find_most_related_prompt(input_command)
  print(gpt_output)
  json_output = json.loads(gpt_output)
  prompt = json_output.get('prompt', '')
  sector = json_output.get('sector', '')
  stock_value_value = json_output.get('stock_value_value', '')
  stock_value_value_list = json_output.get('stock_value_value_list', '')
  date = json_output.get('date', '')
  year = json_output.get('year', '')
  financial_statement_value = json_output.get('financial_statement_value', '')
  condition = json_output.get('condition', '')
  quarter = json_output.get('quarter', '')
  if prompt == '':
    print("No Match: PROMPT\n")
    return "error", ""
  output_command = create_cypher_code(prompt, sector, stock_value_value, stock_value_value_list, date, year, financial_statement_value, condition, quarter)

  if output_command == "":
    return "error", ""
  else:
    driver: Driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'Fin_KG1234'), encrypted=False)
    with driver.session() as session:
      with session.begin_transaction() as tx:
        records = ""
        for row in tx.run(output_command):
          records += str(row) + "\n"
        records = records[:-1]

        column_pattern = r"(\w+)="
        column_names = re.findall(column_pattern, records)
        column_names = list(OrderedDict.fromkeys(column_names))

        data_pattern = r"'([^']+)'|(\d+\.\d+)|(\d+)"
        data = []
        for line in records.split('\n'):
            if line.strip():
                matches = re.findall(data_pattern, line)
                values = [match[0] if match[0] else float(match[1]) if match[1] else int(match[2]) for match in matches]
                data.append(values)
        if len(data) != 0:
          if data[0][0][0] == "'":
            for i in range(len(data)):
              data[i][0] = data[i][0][1:-1]

        for i in range(len(data)):
          data[i][0] = '<a href="https://www.set.or.th/en/market/product/stock/quote/' + data[i][0] + '/price" target="_blank">' + data[i][0] + '</a>'

        if data == []:
          return "error", ""
        df = pd.DataFrame(data, columns=column_names)

        keywords = ["MATCH", "WITH", "WHERE", "RETURN", "LIMIT", "ORDER BY", "DESC", "AS", "AND"]
        pattern_string = r'("[^"]+?")'
        pattern_brackets = r'\(([^:\)]+):'
        pattern_dot = r'\s([a-zA-Z]+?)\.'
        colored_output_command = re.sub(pattern_string, r'<span class="color-custom">\1</span>', output_command)
        colored_output_command = re.sub(pattern_brackets, r'(<span class="color-purple">\1</span>:', colored_output_command)
        colored_output_command = re.sub(pattern_dot, r' <span class="color-purple">\1</span>.', colored_output_command)
        for keyword in keywords:
          colored_output_command = colored_output_command.replace(keyword, f'<span class="color-green">{keyword}</span>')
        colored_output_command = colored_output_command.replace('\n', '<br>')
        return df.style.to_html(index=False), colored_output_command
