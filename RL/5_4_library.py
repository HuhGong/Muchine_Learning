 # 5_4_library.py


# 퀴즈
import re
import requests
# 성남시 해오름도서관에서 책 제목 검색한 결과를 파싱하세요
# 출력: 제목, 저자, 출판사, 발행연도
payload = {
'searchType': 'SIMPLE',
'searchCategory': 'BOOK',
'searchKey': 'ALL',
'searchLibraryArr': 'MH',
'topSearchType': 'BOOK',
'searchKeyword': '고래',
}

response = requests.post(r'https://www.snlib.go.kr/hor/plusSearchResultList.do', params=payload)
result = re.findall(r'<ul class="resultList imageType">(.+?)</ul>', response.text, re.DOTALL)

li = re.findall(r'<li>(.+?)</li', result[0], re.DOTALL)

for item in li:
    tile = re.findall(r'title=(.+?)선택', item, re.DOTALL)
    writer = re.findall(r'')

    print(tile)

