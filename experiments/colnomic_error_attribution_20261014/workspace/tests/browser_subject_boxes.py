"""Optional real-browser regression: python -B tests/browser_subject_boxes.py.

Requires Playwright + Chromium + Pillow. Every write uses a temporary fixture.
"""
import json
import os
import threading

from PIL import Image
from playwright.sync_api import sync_playwright
from test_subject_boxes import SubjectBoxTest
from review_photos import ReviewServer


def main():
    fixture = SubjectBoxTest(); fixture.setUp()
    for row in fixture.store.packages.values():
        Image.new('RGB',(120,80),'white').save(fixture.root / row['design_path'])
    fixture.boxes.save_prelabel('photo_1_1', [15, 10, 100, 65], confidence=0.93)
    server = ReviewServer(('127.0.0.1',0),fixture.store)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    before = fixture.store.package('package_001')[1]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True,args=['--no-sandbox'],
                executable_path=os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
            page = browser.new_page(viewport={'width':1440,'height':1000})
            errors=[];page.on('pageerror',lambda e: errors.append(str(e)))
            page.goto(url);page.locator('.subject-box-button').first.wait_for()
            preview=page.locator('.subject-box-preview[data-photo="photo_1_1"]')
            preview.wait_for(state='visible')
            assert preview.locator('rect').get_attribute('x')=='15'
            def check_preview_alignment():
                error=preview.evaluate('''svg=>{
                    const image=svg.parentElement.querySelector('img'), b=image.getBoundingClientRect();
                    const scale=Math.min(b.width/image.naturalWidth,b.height/image.naturalHeight);
                    const left=b.left+(b.width-image.naturalWidth*scale)/2;
                    const top=b.top+(b.height-image.naturalHeight*scale)/2;
                    const r=svg.firstElementChild.getBoundingClientRect();
                    return Math.max(Math.abs(r.left-(left+15*scale)),Math.abs(r.top-(top+10*scale)),
                        Math.abs(r.width-85*scale),Math.abs(r.height-55*scale));
                }''')
                assert error<1,error
            page.locator('.card[data-photo="photo_1_1"] img').evaluate('(img)=>img.decode()')
            check_preview_alignment()
            page.set_viewport_size({'width':700,'height':1000});check_preview_alignment()
            page.set_viewport_size({'width':1440,'height':1000})
            page.locator('.subject-box-button[data-photo="photo_1_1"]').click()
            page.get_by_text('检查目标和边界，合适就点击“确认此框并下一张”。',exact=True).wait_for()
            page.locator('#box-save').click();page.get_by_text('无需保存，当前框已生效',exact=True).wait_for()
            assert not fixture.boxes.path.exists()
            page.locator('#box-close').click()
            page.locator('.subject-box-button').first.click()
            page.get_by_text('拖动框选目标包装，完成后保存',exact=True).wait_for()
            def pixel(x,y):
                return page.locator('#box-svg').evaluate('(el,p)=>{const q=new DOMPoint(...p).matrixTransform(el.getScreenCTM());return [q.x,q.y];}',[x,y])
            def drag(start,end):
                page.mouse.move(*pixel(*start));page.mouse.down();page.mouse.move(*pixel(*end),steps=5);page.mouse.up()
            drag((100,60),(10,20))  # Reverse drag on an EXIF-rotated image.
            page.locator('#box-save').click();page.get_by_text('已保存',exact=True).wait_for()
            saved=fixture.boxes.get('photo_1_0')
            assert all(abs(a-b)<.1 for a,b in zip(saved['bbox'],[10,20,100,60])),saved
            page.screenshot(path='/tmp/release-subject-box-editor.png')
            # A competing window must not overwrite the first window's newer edit.
            second=browser.new_page(viewport={'width':1440,'height':1000});second.goto(url)
            second.locator('.subject-box-button').first.click()
            second.get_by_text('检查目标和边界，合适就点击“确认此框并下一张”。',exact=True).wait_for()
            drag((30,30),(35,35))
            page.locator('#box-save').click();page.get_by_text('已保存',exact=True).wait_for()
            moved=fixture.boxes.get('photo_1_0')['bbox']
            assert all(abs(a-b)<.1 for a,b in zip(moved,[15,25,105,65])),moved
            second.locator('#box-note').fill('stale edit');second.locator('#box-save').click()
            second.get_by_text('This box changed in another window. Reopen the photo before saving.',exact=True).wait_for()
            second.on('dialog',lambda d:d.accept());second.close()
            # Resize a corner, persist, then reopen after a full page reload.
            drag((105,65),(110,70));page.locator('#box-save').click();page.get_by_text('已保存',exact=True).wait_for()
            page.locator('#box-close').click();page.reload();page.locator('.subject-box-button').first.click()
            page.get_by_text('检查目标和边界，合适就点击“确认此框并下一张”。',exact=True).wait_for()
            assert all(abs(a-b)<.1 for a,b in zip(fixture.boxes.get('photo_1_0')['bbox'],[15,25,110,70]))
            page.locator('#box-unclear').check();page.locator('#box-save-next').click()
            page.locator('#box-title').filter(has_text='photo_1_1').wait_for()
            page.get_by_text('检查目标和边界，合适就点击“确认此框并下一张”。',exact=True).wait_for()
            assert fixture.boxes.get('photo_1_0')['status']=='unclear'
            assert fixture.boxes.get('photo_1_1')['status']=='accepted'
            assert page.locator('#box-svg').get_attribute('data-status')=='accepted'
            drag((30,30),(35,35))
            page.locator('#box-save').click();page.get_by_text('已保存',exact=True).wait_for()
            assert fixture.boxes.get('photo_1_1')['status']=='confirmed'
            assert all(abs(a-b)<.1 for a,b in zip(fixture.boxes.get('photo_1_1')['bbox'],[20,15,105,70]))
            assert abs(float(preview.locator('rect').get_attribute('x'))-20)<.1
            page.locator('#box-use-prelabel').click()
            page.locator('#box-save').click();page.get_by_text('已保存',exact=True).wait_for()
            assert fixture.boxes.get('photo_1_1')['bbox']==[15,10,100,65]
            page.locator('#box-prev').click();page.get_by_text('拖动框选目标包装，完成后保存',exact=True).wait_for()
            page.locator('#box-clear').click();page.locator('#box-save').click();page.get_by_text('已保存',exact=True).wait_for()
            assert fixture.boxes.get('photo_1_0')['status']=='pending'
            # Confirm an unchanged automatic box and count only the review scope.
            data=fixture.boxes.read();del data['records']['photo_1_1']
            fixture.boxes.path.write_text(json.dumps(data))
            page.locator('#box-close').click();page.reload()
            page.locator('.subject-box-button[data-photo="photo_1_1"]').wait_for()
            page.evaluate("SubjectBoxEditor.setScope(new Set(['photo_1_1']))")
            page.locator('.subject-box-button[data-photo="photo_1_1"]').click()
            page.get_by_text('检查目标和边界，合适就点击“确认此框并下一张”。',exact=True).wait_for()
            assert fixture.boxes.get('photo_1_1')['status']=='accepted'
            page.locator('#box-confirm-next').click()
            page.locator('#box-progress').filter(has_text='人工已校验 1 / 1').wait_for()
            confirmed=fixture.boxes.get('photo_1_1')
            assert confirmed['status']=='confirmed' and confirmed['bbox']==[15,10,100,65]
            # The original scope omits added photos; pending navigation skips
            # confirmed boxes even when their scene labels are still unreviewed.
            receipt=fixture.root/'review/supplement_fixture/receipt.json'
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps({'photo_ids':['photo_2_0']}))
            (fixture.root/'release.json').write_text(json.dumps({'supplement':'review/supplement_fixture/receipt.json'}))
            for pid in ('photo_1_0','photo_1_2','photo_2_0'):
                fixture.boxes.save_prelabel(pid,[15,10,100,65])
            page.locator('#box-close').click()
            page.goto(url+'/?scope=original&filter=boxes_pending')
            page.locator('.subject-box-button[data-photo="photo_1_0"]').wait_for()
            assert page.locator('#scope').input_value()=='original'
            assert page.locator('#grid .card:visible').count()==2
            page.locator('#box-progress').filter(has_text='人工已校验 1 / 3').wait_for()
            page.locator('.subject-box-button[data-photo="photo_1_0"]').click()
            page.locator('#box-use-prelabel').click()
            page.locator('#box-confirm-next').click()
            page.locator('#box-title').filter(has_text='photo_1_2').wait_for()
            page.get_by_text('检查目标和边界，合适就点击“确认此框并下一张”。',exact=True).wait_for()
            page.keyboard.press('Control+Enter')
            page.locator('#box-progress').filter(has_text='人工已校验 3 / 3').wait_for()
            page.get_by_text('当前范围的主体框已全部人工校验。',exact=True).wait_for()
            assert fixture.boxes.get('photo_2_0')['status']=='accepted'
            assert page.locator('#grid .card:visible').count()==0
            page.locator('#box-close').click()
            page.goto(url+'/?scope=all&filter=boxes_pending')
            page.locator('.subject-box-button[data-photo="photo_2_0"]').wait_for()
            assert page.locator('#grid .card:visible').count()==1
            page.locator('.subject-box-button[data-photo="photo_2_0"]').click()
            page.get_by_text('检查目标和边界，合适就点击“确认此框并下一张”。',exact=True).wait_for()
            page.locator('#box-confirm-next').click()
            page.locator('#box-progress').filter(has_text='人工已校验 4 / 4').wait_for()
            assert fixture.store.package('package_001')[1]==before
            assert not errors,errors
            browser.close()
            print(json.dumps({'browser':'chromium','reverse_draw':True,'move_resize':True,
                'save_reload':True,'conflict':True,'unclear_clear_next':True,'scene_labels_unchanged':True,
                'default_boxes_without_confirmation':True,'responsive_card_boxes':True,
                'saved_edit_updates_card':True,'prelabel_load_edit_restore_confirm':True,
                'unchanged_box_human_confirmation':True,'scoped_review_progress':True,
                'pending_box_queue_skips_confirmed':True,'original_scope':True,
                'keyboard_confirmation':True,'scene_labels_independent_of_box_filter':True}))
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2);fixture.tearDown()


if __name__=='__main__':
    main()
