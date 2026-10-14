"""Exercise review in Chromium; every annotation write uses a temporary fixture."""
import json
import os
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright
from test_error_analysis import ErrorAnalysisTest
from review_errors import ReviewServer


def main():
    fixture = ErrorAnalysisTest()
    fixture.setUp()
    server = ReviewServer(('127.0.0.1', 0), fixture.store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'],
                executable_path=os.environ.get('DRUGPACK_CHROMIUM_EXECUTABLE'))
            page = browser.new_page(viewport={'width': 1550, 'height': 1080})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            page.locator('#case:not([hidden])').wait_for()
            assert page.locator('#config').input_value() == 'F03'
            assert page.locator('#predictions tr').count() == 35
            page.locator('#search').fill('photo0')
            page.locator('#search').press('Enter')
            key = fixture.case_key()
            page.wait_for_function('(key) => document.querySelector("#case-title").textContent === key', arg=key)
            assert page.locator('.main-select[data-config="F03"]').input_value() == 'E4'
            assert page.locator('#images img').count() == 3
            assert page.locator('#processed-images img').count() == 2
            page.locator('#draft').click()
            page.get_by_text('草稿已保存，仍为待确认。', exact=True).wait_for()
            assert not fixture.store.annotations()[key]['confirmed']
            page.locator('#confusion').select_option('C2')
            page.locator('#faces input[value="back"]').check()
            page.locator('#clue-presence').select_option('absent')
            assert page.locator('#clue-readability').input_value() == 'not_present'
            page.locator('#causes input[value="反光"]').check()
            page.locator('#config-causes input[data-config="D03"][value="SAM漏分割"]').check()
            page.locator('#shared-issue').select_option('E0')
            assert all(e == 'E0' for e in page.locator('.main-select').evaluate_all('(elements) => elements.map(e => e.value)'))
            page.locator('#note').fill('背面无规格信息，身份歧义；SAM 输入遗漏区分区。')
            page.locator('#save-next').click()
            page.get_by_text('已确认并映射到相关配置。', exact=True).wait_for()
            page.locator('#empty:not([hidden])').wait_for()
            label = fixture.store.annotations()[key]
            assert label['confirmed'] and label['confusion'] == 'C2'
            assert label['config_causes'] == {'D03': ['SAM漏分割']}
            assert set(label['main_by_config'].values()) == {'E0'}
            page.locator('#status').select_option('confirmed')
            page.locator('#case:not([hidden])').wait_for()
            assert page.locator('#confusion').input_value() == 'C2'
            assert page.locator('#faces input[value="back"]').is_checked()
            assert page.locator('.main-select[data-config="F03"]').input_value() == 'E0'
            page.reload()
            page.locator('#case:not([hidden])').wait_for()
            assert page.locator('#status').input_value() == 'confirmed'
            assert page.locator('#search').input_value() == 'photo0'
            page.wait_for_function('(key) => document.querySelector("#case-title").textContent === key', arg=key)
            assert page.locator('#confusion').input_value() == 'C2'
            page.locator('#export').click()
            page.get_by_text('全部报告已更新：results/error_analysis/', exact=True).wait_for()
            page.set_viewport_size({'width': 700, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Horizontal overflow'
            page.screenshot(path='/tmp/drugpack-error-review-mobile.png', full_page=True)
            assert not errors, errors
            browser.close()
        for path, original in fixture.original.items():
            assert path.read_bytes() == original, str(path)
        print(json.dumps(dict(browser='Chromium', unique_pair_maps_to_all_configurations=True,
            draft_then_confirm=True, processing_reasons_scoped=True, filters_and_persistence=True,
            responsive=True, original_inventories_unchanged=True)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        fixture.tearDown()


if __name__ == '__main__':
    main()
