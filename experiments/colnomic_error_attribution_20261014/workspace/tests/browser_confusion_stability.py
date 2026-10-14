"""Browser regression for scroll, stable card identity, queues and progress.

Uses a temporary multi-page catalog; never writes real annotations.
"""
import hashlib
import json
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright, expect
from test_confusion_review import ConfusionReviewTest
from test_photo_review import csv_file
from photo_metadata import read_csv, write_json
from confusion_review import ConfusionReview, inventory_signature
from review_photos import ReviewServer


def main():
    f=ConfusionReviewTest();f.setUp()
    refs=read_csv(f.root/'lists/references.csv');catalog=f.review.catalog()
    for i in range(4,46):
        rid=f'reference_{i:06d}';path=f.root/f'references/images/{rid}.jpg'
        Image.new('RGB',(160,100),(i*5%256,i*13%256,i*23%256)).save(path)
        refs.append(dict(reference_id=rid,row=str(i),path=path.relative_to(f.root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity=f'Candidate {i}',category='human/otc',original_filename=rid+'.jpg'))
        catalog['references'][rid]=dict(title=f'Candidate {i}',ocr_preview='Example medicine',search_text='Example medicine')
        catalog['packages']['package_001'].append(dict(reference_id=rid,proposed_categories=['text_similar'],evidence=['Synthetic review fixture']))
    csv_file(f.root/'lists/references.csv',list(refs[0]),refs)
    catalog['inventory']=inventory_signature(f.root);write_json(f.review.candidate_path,catalog)
    f.review=ConfusionReview(f.store)
    server=ReviewServer(('127.0.0.1',0),f.store);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/confusion'
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1450,'height':680});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(url);page.locator('.candidate-card').first.wait_for()
            expect(page.locator('#package-review-progress')).to_contain_text('待标 44')
            page.locator('#review-tabs [data-filter="pending"]').click();page.locator('#page-next').click()
            expect(page.locator('#page-position')).to_have_text('2 / 3')
            card=page.locator('.candidate-card').nth(5);rid=card.get_attribute('data-reference')
            card.get_by_role('button',name='并排审核',exact=True).click()
            page.locator('#pair-dialog[open]').wait_for()
            snapshot=page.evaluate('''() => {
                window.stableCard=document.querySelectorAll('.candidate-card')[5];
                window.stableImage=window.stableCard.querySelector('img');
                const r=window.stableCard.getBoundingClientRect();
                return {main:document.querySelector('main').scrollTop, sidebar:document.querySelector('.packages-panel').scrollTop,
                    anchor:document.querySelector('.anchor-panel').scrollTop, window:scrollY, y:r.top, height:r.height};
            }''')
            assert snapshot['main']>0,snapshot
            page.locator('#pair-note').fill('A long saved note that must not resize the review card or move the page.')
            page.locator('[data-decision="not_confusing"]').click()
            expect(page.locator('#pair-message')).to_have_text('已保存 · 不混淆')
            page.locator('#pair-close').click()
            expect(page.locator(f'.candidate-card[data-reference="{rid}"] .review-badge')).to_have_text('✓ 已标 · 不混淆')
            after=page.evaluate('''() => {
                const r=window.stableCard.getBoundingClientRect();
                return {main:document.querySelector('main').scrollTop,sidebar:document.querySelector('.packages-panel').scrollTop,
                    anchor:document.querySelector('.anchor-panel').scrollTop,window:scrollY,y:r.top,height:r.height,
                    sameCard:document.querySelectorAll('.candidate-card')[5]===window.stableCard,
                    sameImage:window.stableCard.querySelector('img')===window.stableImage};
            }''')
            assert after['sameCard'] and after['sameImage'],after
            for k in snapshot:assert abs(snapshot[k]-after[k])<1,(k,snapshot,after)
            assert page.locator('.candidate-card').count()==18
            expect(page.locator('#review-tabs [data-filter="pending"]')).to_have_text('待标注 43')
            expect(page.locator('#package-review-progress')).to_contain_text('已定 1/44')
            # The current pending queue stays in place until the user explicitly reapplies a filter.
            page.locator('#review-tabs [data-filter="pending"]').click()
            assert page.locator(f'.candidate-card[data-reference="{rid}"]').count()==0
            page.locator('#review-tabs [data-filter="finalized"]').click()
            expect(page.locator('.candidate-card')).to_have_count(1)
            expect(page.locator('.candidate-card')).to_have_attribute('data-reference',rid)
            # A reload restores filters, the second page and all three scroll panels.
            page.locator('#review-tabs [data-filter="all"]').click();page.locator('#page-next').click()
            expect(page.locator('#page-position')).to_have_text('2 / 3')
            page.evaluate("() => {document.querySelector('main').scrollTop=530;document.querySelector('.packages-panel').scrollTop=20;document.querySelector('.anchor-panel').scrollTop=80;}")
            saved=page.evaluate("() => ({main:document.querySelector('main').scrollTop,sidebar:document.querySelector('.packages-panel').scrollTop,anchor:document.querySelector('.anchor-panel').scrollTop})")
            page.reload();expect(page.locator('#page-position')).to_have_text('2 / 3')
            page.wait_for_function("s => Object.entries(s).every(([k,v])=>Math.abs(document.querySelector(k==='main'?'main':k==='sidebar'?'.packages-panel':'.anchor-panel').scrollTop-v)<1)",arg=saved)
            assert page.locator('#scope').input_value()=='all'
            # Consecutive automatic-next operations follow IDs and skip finished items.
            candidates=page.locator('.candidate-card').evaluate_all('(xs)=>xs.map(x=>x.dataset.reference)')
            target=candidates[1]
            page.locator(f'.candidate-card[data-reference="{target}"] .compare-button').click()
            page.locator('#pair-auto-next').check()
            page.locator('[data-decision="not_confusing"]').click()
            expect(page.locator('#pair-right-label')).to_contain_text(candidates[2])
            page.locator('[data-decision="not_confusing"]').click()
            expect(page.locator('#pair-right-label')).to_contain_text(candidates[3])
            page.locator('#pair-auto-next').uncheck();page.locator('[data-decision="unsure"]').click()
            expect(page.locator('#pair-message')).to_have_text('已保存 · 暂不确定')
            page.locator('#pair-close').click()
            expect(page.locator('#review-tabs [data-filter="unsure"]')).to_have_text('待复核 1')
            page.locator('#review-tabs [data-filter="unsure"]').click();expect(page.locator('.candidate-card')).to_have_count(1)
            expect(page.locator('.candidate-card .review-badge')).to_have_text('? 待复核 · 暂不确定')
            page.set_viewport_size({'width':720,'height':1000})
            assert page.evaluate('() => document.documentElement.scrollWidth<=innerWidth')
            assert not errors,errors
            browser.close()
        print(json.dumps(dict(scroll_delta_pixels=0,card_and_image_nodes_preserved=True,
            pending_card_pinned_until_refilter=True,accurate_progress=True,saved_and_pending_filters=True,
            reload_restores_page_filters_scroll=True,consecutive_next_skips_saved=True,unsure_separate=True)))
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2);f.tearDown()


if __name__=='__main__':main()
