import Levenshtein as levenshtein
import re
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from neo4j import GraphDatabase, Driver
import pandas as pd
from collections import OrderedDict

PROMPT = [
  # 8
  "get average, open, close on 2023/10/11",
  # 7
  "highest totalAssets in 2023",
  # 0
  "for setcor ENERG order according to totalAssets based on all financial statements",
  "sector energ, total assets, all reports",
  # 1
  "for sector ENERG order according to totalAssets based on latest financial statements",
  "sector energ, total assets, latest reports"
  # 2
  "is 1 or more",
  "is more than 1",
  "is over 1",
  "is above 1",
  "is 1 or less",
  "is under 1",
  # 3
  "safe stock order",
  # 4
  "big change in stock price on 2023/10/11",
  # 5
  "high average stock price on 2023/10/11",
  "high average price on 2023/10/11",
  "high average stock price on 2023/10/11(open, close)",
  "high average stock price on 2023/10/11 with open, close"
  # 6
  "for sector ENERG high average stock price on 2023/10/11",
  "sector food high average value on 2023/10/11",
  "sector energ high average on 2023/10/11 with open, close",
  "sector energ high average on 2023/10/11(open, close)"
]

PROMPT_ID = [
  8,
  7,
  0,
  0,
  1,
  1,
  2,
  2,
  2,
  2,
  2,
  2,
  3,
  4,
  5,
  5,
  5,
  6,
  6,
  6,
  6
]

# https://media.set.or.th/set/Documents/2022/Oct/05_1_Company_Fundamental_Specification.pdf

STOCK_VALUE_VALUE = [
  "average",
  "close",
  "high",
  "low",
  "open",
  "prior",
  "totalVolume"
]

# if no mention, select 'Quarter', and select 'Accum' if specified
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

SECTOR = ["ICT", "TRANS", "PROP", "ENERG", "BANK", "HELTH", "FOOD", "TOURISM", "COMM", "ETRON", "PETRO", "FIN", "CONMAT", "PKG", "INSUR"]

CONDITION = [
  "or more",
  "or over",
  "or above",

  "more than",
  "larger than",
  "greater than",
  "bigger than",
  "over",
  "above",

  "or less",
  "or under",
  "or below",

  "less than",
  "under",
  "below",
  "smaller than"
]

STANDARD_CONDITION = [
  ">=",
  ">=",
  ">=",

  ">",
  ">",
  ">",
  ">",
  ">",
  ">",

  "<=",
  "<=",
  "<=",

  "<",
  "<",
  "<",
  "<"
]

# Caluculate cos related ratio of two texts
def calc_cos_text(text1,text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors)
    max_similarity = np.max(np.triu(similarity, k=1))
    return max_similarity

# Caluculate cos related ratio of two words
def calc_cos_word(word1, word2):
  return levenshtein.jaro_winkler(word1, word2)

# Find the most related prompt
def find_most_related_prompt(text):
  max_cos = 0
  max_cos_index = -1

  if "safe" in text:
    return 3

  for i in range(len(PROMPT)):
    cos = calc_cos_text(text, PROMPT[i])
    if cos > max_cos:
      max_cos = cos
      max_cos_index = PROMPT_ID[i]
  return max_cos_index

def find_sector(text):
  max_cos = 0
  most_similar_sector = ""
  split_text = text.split()
  for w in split_text:
    for s in SECTOR:
      cos = calc_cos_word(w, s)
      if cos > max_cos:
        max_cos = cos
        most_similar_sector = s
      cos = calc_cos_word(w, s.lower())
      if cos > max_cos:
        max_cos = cos
        most_similar_sector = s

  if most_similar_sector == "":
    print("No Match: SECTOR")
    return "ERROR"
  if most_similar_sector == "COMM":
    if "information" in text:
      most_similar_sector = "ICT"
  elif most_similar_sector == "PETRO":
    if "construction" in text:
      most_similar_sector = "CONMAT"
  return most_similar_sector

def find_financial_statement_value(text):
  max_cos = 0
  most_similar_word = ""
  split_text = text.split()
  for w in split_text:
    for f in FINANCIAL_STATEMENT_VALUE:
      if w != 'total':
        if w.replace(',', '') == f:
          return f
        elif w.replace(',', '') in f or w.replace(',', '') in f.lower():
          if f == 'totalAssets':
            if 'fixed' in text:
              return 'fixedAssetTurnover'
            elif 'turnover' in text:
              return 'totalAssetsTurnover'
            else:
              return 'totalAssets'
          elif f == 'totalRevenueQuarter':
            if 'accum' in text:
              return 'totalRevenueAccum'
            else:
              return 'totalRevenueQuarter'
          elif f == 'ebitQuarter':
              if 'accum' in text:
                return 'ebitAccum'
              else:
                return 'ebitQuarter'
          elif f == 'netProfitQuarter':
              if 'accum' in text:
                return 'netProfitAccum'
              else:
                return 'netProfitQuarter'
          elif f == 'epsQuarter':
              if 'accum' in text:
                return 'epsAccum'
              else:
                return 'epsQuarter'
          elif f == 'operatingCacheFlow':
              if 'financ' in text:
                return 'financingCacheFlow'
              elif 'invest' in text:
                return 'investingCacheFlow'
              else:
                return 'operatingCacheFlow'
          elif f == 'shareholderEquity':
              if 'shareholder' in text:
                return 'shareholderEquity'
              else:
                return 'totalEquity'
          else:
              return f

        cos = calc_cos_word(w, f)
        if cos > max_cos:
          max_cos = cos
          most_similar_word = f

  if most_similar_word == "":
    print("No Match: FINANCIAL_STATEMENT_VALUE")
    return "ERROR"
  return most_similar_word

def find_cond(text):
  most_similar_index = -1
  for i in range(len(CONDITION)):
    if CONDITION[i] in text:
      most_similar_index = i
      break

  if most_similar_index == -1:
    print("No Match: CONDITION")
    return "ERROR"

  financial_statement_value = find_financial_statement_value(text)
  num = str(re.findall(r'\d+', text)[0])

  standard_condition = financial_statement_value + ' ' + STANDARD_CONDITION[i] + ' ' + num
  return standard_condition

# Find yyyy/mm/dd, yyyy/mm/d, yyyy/m/dd, yyyy/m/d, yyyy-mm-dd, yyyy-mm-d, yyyy-m-dd, yyyy-m-d
def find_date(text):
  pattern1 = r'\b\d{4}/(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])\b'
  pattern2 = r'\b\d{4}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12][0-9]|3[01])\b'

  match1 = re.findall(pattern1, text)
  match2 = re.findall(pattern2, text)

  pattern = ''
  if match1 == [] and match2 == []:
    print("No Match: DATE")
    return "ERROR"
  elif match1 == []:
    pattern = str(match2[0]).replace('-', '/')
  else:
    pattern = str(match1[0])
  return pattern

def find_stock_value_value(text):
  max_cos = 0
  most_similar_word = ""
  split_text = text.split()
  for w in split_text:
    for v in STOCK_VALUE_VALUE:
      if w != 'high':
        if w.replace(',', '') == v:
          return v
      cos = calc_cos_word(w, v)
      if cos > max_cos:
        max_cos = cos
        most_similar_word = v

  if most_similar_word == "":
    print("No Match: STOCK_VALUE_VALUE")
    return "ERROR"
  return most_similar_word

def find_year(text):
  pattern = r'\b\d{4}\b'
  return re.findall(pattern, text)[0]

def get_stock_value_value_list(text):
  output = []
  split_text = text.split()
  for w in split_text:
    for i in range(len(STOCK_VALUE_VALUE)):
      if w in STOCK_VALUE_VALUE[i] or w[:-1] in STOCK_VALUE_VALUE[i] or w[1:-1] in STOCK_VALUE_VALUE[i]:
        output.append(STOCK_VALUE_VALUE[i])
        break
  output = list(OrderedDict.fromkeys(output))
  if output == []:
    print("No Match: STOCK_VALUE_VALUE")
    return "ERROR"
  return output

def create_cypher_code(prompt_id, text):
  output_command = ""
  if prompt_id == 0:
    sector = find_sector(text)
    financial_statement_value = find_financial_statement_value(text)
    if sector == "ERROR" or financial_statement_value == "ERROR":
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
    sector = find_sector(text)
    financial_statement_value = find_financial_statement_value(text)
    if sector == "ERROR" or financial_statement_value == "ERROR":
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
    cond = find_cond(text)
    if cond == "ERROR":
      return ""
    output_command = "MATCH (c:FinancialStatement)\n" + \
      "WITH 10 * c.year + c.quarter AS yearQuarterSum\n" + \
      "WITH COLLECT(yearQuarterSum) AS maxSums\n" + \
      "WITH MAX(maxSums)[0] AS maxSum\n" + \
      "MATCH (a:Stock)<-[:STOCK]-(:Company)-[:REPORT]->(b:FinancialStatement)\n" + \
      "WHERE b." + cond + " AND 10 * b.year + b.quarter = maxSum\n" + \
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
    today_str = find_date(text)
    if today_str == "ERROR":
      return ""
    today = datetime.datetime.strptime(today_str, '%Y/%m/%d')
    yesterday = today + datetime.timedelta(days=-1)
    yesterday = "{:%Y/%m/%d}".format(yesterday).replace("/0", "/")
    output_command = 'MATCH (todayPrice:StockValue {date: "' + today_str + '"})<-[:STOCK_VALUE]-(stock:Stock)\n' + \
      'MATCH (stock)-[:STOCK_VALUE]->(yesterdayPrice:StockValue {date: "' + yesterday + '"})\n' + \
      'WITH stock.symbol AS symbol, todayPrice, yesterdayPrice\n' + \
      'RETURN symbol, todayPrice.average - yesterdayPrice.average AS priceDifference\n' + \
      'ORDER BY priceDifference DESC'
  elif prompt_id == 5:
    today_str = find_date(text)
    stock_value_value = find_stock_value_value(text)
    if today_str == "ERROR" or stock_value_value == "ERROR":
      return ""
    if "(" in text or 'with' in text:
      stock_value_value_list = get_stock_value_value_list(text)
      stock_value_value_list.remove(stock_value_value)
      if stock_value_value_list == "ERROR":
        return ""
      output_command = 'MATCH (a:Stock)-[r:STOCK_VALUE {date: "' + today_str + '"}]->(b:StockValue)\n' + \
        "WITH a.symbol AS symbol, b." + stock_value_value + " AS " + stock_value_value + ", "
      for s in stock_value_value_list:
        output_command += "b." + s + " AS " + s + ", "
      output_command = output_command[:-2]
      output_command += "\nRETURN symbol, " + stock_value_value + ", "
      for s in stock_value_value_list:
        output_command += s + ", "
      output_command = output_command[:-2] + "\nORDER BY " + stock_value_value + " DESC"
    else:
      output_command = 'MATCH (a:Stock)-[r:STOCK_VALUE {date: "' + today_str + '"}]->(b:StockValue)\n' + \
        "WITH a.symbol AS symbol, b." + stock_value_value + " AS " + stock_value_value + "\n" + \
        "RETURN symbol, " + stock_value_value + "\n" + \
        "ORDER BY " + stock_value_value + " DESC"
  elif prompt_id == 6:
    sector = find_sector(text)
    today_str = find_date(text)
    stock_value_value = find_stock_value_value(text)
    if sector == "ERROR" or today_str == "ERROR" or stock_value_value == "ERROR":
      return ""
    if "(" in text or "with" in text:
      stock_value_value_list = get_stock_value_value_list(text)
      stock_value_value_list.remove(stock_value_value)
      if stock_value_value_list == "ERROR":
        return ""
      output_command = "MATCH (:StandardIndustrialClassification {sectorCode: '" + sector + "'})<-[:BELONGING_SECTOR]-(:Company)-[:STOCK]->(c:Stock)-[:STOCK_VALUE {date: '" + today_str + "'}]->(d:StockValue)\n" + \
        "WITH c.symbol AS symbol, d." + stock_value_value + " AS " + stock_value_value + ", "
      for s in stock_value_value_list:
        output_command += "d." + s + " AS " + s + ", "
      output_command = output_command[:-2] + "\nRETURN symbol, " + stock_value_value + ", "
      for s in stock_value_value_list:
        output_command += s + ", "
      output_command = output_command[:-2] + "\nORDER BY " + stock_value_value + " DESC"
    else:
      output_command = "MATCH (:StandardIndustrialClassification {sectorCode: '" + sector + "'})<-[:BELONGING_SECTOR]-(:Company)-[:STOCK]->(c:Stock)-[:STOCK_VALUE {date: '" + today_str + "'}]->(d:StockValue)\n" + \
        "WITH c.symbol AS symbol, d." + stock_value_value + " AS " + stock_value_value + "\n" + \
        "RETURN symbol, " + stock_value_value + "\n" + \
        "ORDER BY " + stock_value_value + " DESC"
  elif prompt_id == 7:
    year = find_year(text)
    financial_statement_value = find_financial_statement_value(text)
    if year == "ERROR" or financial_statement_value == "ERROR":
      return ""
    output_command = "MATCH (a:Stock)<-[:STOCK]-(:Company)-[:REPORT]->(b:FinancialStatement)\n" + \
      "WHERE " + "b.year = " + year + "\n" + \
      "WITH a.symbol AS symbol, b.year AS year, b.quarter AS quarter, b." + financial_statement_value + " AS " + financial_statement_value + "\n" + \
      "RETURN symbol, year, quarter, " + financial_statement_value
  elif prompt_id == 8:
    date = find_date(text)
    stock_value_value_list = get_stock_value_value_list(text)
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

  return output_command

def main(input_command):
  if input_command == "":
    return "error", ""

  prompt_id = find_most_related_prompt(input_command)
  if prompt_id < 0:
    print("No Match: PROMPT\n")
    return "error", ""
  output_command = create_cypher_code(prompt_id, input_command)

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
