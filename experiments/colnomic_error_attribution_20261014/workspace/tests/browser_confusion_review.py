"""Optional Chromium integration test. All writes target a temporary dataset."""
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright
from test_confusion_review import ConfusionReviewTest
from review_photos import ReviewServer


def main():
    fixture=ConfusionReviewTest();fixture.setUp()
    server=ReviewServer(('127.0.0.1',0),fixture.store)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1500,'height':1050});errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url);page.get_by_role('link',name='易混淆包装 · 验证子集 →').click()
            card=page.locator('.candidate-card[data-reference="reference_000001"]');card.wait_for()
            assert page.locator('#scope').input_value()=='all'
            page.locator('.candidate-card[data-reference="reference_000002"]').wait_for()
            assert '0 个包装 / 0 张实拍' in page.locator('#summary').inner_text()
            card.locator('.compare-button').click()
            page.locator('#pair-dialog[open]').wait_for()
            assert page.locator('#pair-categories input[value="strength_count"]').is_checked()
            page.locator('#pair-note').fill('人工核实：数字规格不同')
            page.locator('[data-decision="confusing"]').click()
            page.get_by_text('已保存 · 已确认混淆',exact=True).wait_for()
            page.wait_for_function("() => document.querySelector('#summary').textContent.includes('2 个包装 / 4 张实拍')")
            page.locator('#pair-close').click()
            page.reload();card.wait_for();assert card.get_attribute('data-status')=='confusing'
            page.locator('[data-package="package_002"]').click()
            reverse=page.locator('.candidate-card[data-reference="reference_000000"]');reverse.wait_for()
            assert reverse.get_attribute('data-status')=='confusing'
            page.locator('[data-package="package_001"]').click();card.wait_for()
            card.locator('.compare-button').click();page.locator('[data-decision="not_confusing"]').click()
            page.get_by_text('已保存 · 不混淆',exact=True).wait_for()
            page.wait_for_function("() => document.querySelector('#summary').textContent.includes('0 个包装 / 0 张实拍')")
            page.locator('#pair-close').click();page.locator('#scope').select_option('other')
            other=page.locator('.candidate-card[data-reference="reference_000002"]');other.wait_for()
            other.get_by_role('button',name='加入待补拍').click()
            page.wait_for_function("() => document.querySelector('#summary').textContent.includes('待补拍 1 个设计')")
            assert fixture.review.summary()['counts']['photos']==0
            other.locator('.compare-button').click();page.locator('[data-decision="confusing"]').click()
            page.get_by_text('已保存 · 已确认混淆',exact=True).wait_for()
            page.wait_for_function("() => document.querySelector('#summary').textContent.includes('1 个包装 / 3 张实拍')")
            page.locator('#pair-close').click()
            page.locator('#package-selection').select_option('exclude');page.locator('#package-reviewed').check()
            page.locator('#package-note').fill('暂不纳入，保留关系');page.locator('#package-save').click()
            page.get_by_text('已保存包装选择。',exact=True).wait_for()
            assert fixture.review.summary()['counts']['photos']==0
            page.reload();other.wait_for();assert page.locator('#scope').input_value()=='other';assert page.locator('#package-selection').input_value()=='exclude'
            assert page.locator('#package-reviewed').is_checked()
            page.locator('#mode').select_option('gallery');page.locator('#scope').select_option('all')
            page.locator('#design-search').fill('reference_000003')
            page.locator('.candidate-card[data-reference="reference_000000"]').wait_for()
            page.wait_for_function("() => document.querySelectorAll('.candidate-card').length===1")
            assert page.locator('.candidate-card').get_attribute('data-reference')=='reference_000000'
            with page.expect_download() as download_info:page.locator('#export').click()
            data=json.loads(Path(download_info.value.path()).read_text())
            assert data['validation_subset']['additional_design_reference_ids']==['reference_000002']
            assert not data['validation_subset']['photo_ids']
            page.set_viewport_size({'width':720,'height':1000})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'horizontal overflow'
            page.screenshot(path='/tmp/release-confusion-review-mobile.png',full_page=True)
            assert not errors,errors
            browser.close()
        for path,original in fixture.original.items():assert path.read_bytes()==original,str(path)
        print(json.dumps(dict(browser='Chromium',confirm_reject_persist=True,symmetric_pairs=True,
            shortlist_does_not_add_queries=True,manual_exclude=True,gallery_alias_search=True,
            metadata_download=True,responsive=True,original_lists_and_labels_unchanged=True)))
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2);fixture.tearDown()


if __name__=='__main__':main()
