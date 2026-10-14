"""Manual scene-review regression; all writes use a temporary synthetic dataset.

Run with a Python environment containing Pillow, Playwright and Chromium.
"""
import json
import os
import threading

from PIL import Image
from playwright.sync_api import expect, sync_playwright
from test_photo_review import PhotoReviewTest, csv_file
from photo_metadata import PhotoStore, read_csv, sha, write_json
from review_photos import ReviewServer


def main():
    fixture = PhotoReviewTest()
    fixture.setUp()
    for row in fixture.store.photos.values():
        Image.new('RGB', (240, 180), 'white').save(fixture.root / row['path'])
        row['sha256'] = sha((fixture.root / row['path']).read_bytes())
    for path in (fixture.root / 'lists').rglob('*.csv'):
        if path.parent.name == 'active' or 'active' in path.parts:
            continue
        rows = read_csv(path)
        if rows and 'photo_id' in rows[0]:
            for row in rows:
                row['sha256'] = fixture.store.photos[row['photo_id']]['sha256']
            csv_file(path, list(rows[0]), rows)
    for row in fixture.store.packages.values():
        Image.new('RGB', (180, 100), 'lightblue').save(fixture.root / row['design_path'])
    write_json(fixture.root / 'release.json', dict(supplement='review/supplements/fixture/import.json'))
    write_json(fixture.root / 'review/supplements/fixture/import.json',
               dict(photo_ids=['photo_1_1', 'photo_1_2', 'photo_2_0']))
    fixture.store = PhotoStore(fixture.root)
    fixture.store.export_active()
    original = fixture.store.records()['photo_1_0']
    formal = (fixture.root / 'lists/photos.csv').read_bytes()
    server = ReviewServer(('127.0.0.1', 0), fixture.store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'],
                executable_path=os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'))
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            expect(page.locator('#scope')).to_have_value('supplement')
            expect(page.locator('.card')).to_have_count(2)
            expect(page.locator('#review-text')).to_contain_text('0 / 3')
            expect(page.locator('#occlusion-text')).to_contain_text('0 / 3')
            assert not any(row['reviewed'] for row in fixture.store.records().values())

            card = page.locator('.card[data-photo="photo_1_1"]')
            reviewed = card.locator('[data-field="reviewed"]')
            occlusion = card.locator('[data-field="occlusion"]')
            expect(occlusion).to_have_value('')
            image = card.locator('.photo img').element_handle()
            reviewed.check()
            expect(page.locator('#review-text')).to_contain_text('0 / 3')  # Saved progress only.
            page.locator('#save').click()
            expect(page.locator('#notice')).to_contain_text('已保存到 metadata')
            assert fixture.store.records()['photo_1_1']['reviewed']
            assert 'occlusion' not in fixture.store.records()['photo_1_1']['type']
            expect(page.locator('#review-text')).to_contain_text('1 / 3')
            assert card.locator('.photo img').evaluate('(node, old) => node === old', image)

            occlusion.select_option('foreground')
            expect(reviewed).not_to_be_checked()
            expect(card.locator('[data-field="background"]')).to_have_value('single_object')
            reviewed.check()
            page.locator('#save').click()
            expect(page.locator('#notice')).to_contain_text('已保存到 metadata')
            expect(page.locator('#occlusion-text')).to_contain_text('1 / 3')
            assert fixture.store.records()['photo_1_1']['type']['occlusion'] == 'foreground'

            card.locator('[data-field="placement"]').select_option('handheld')
            expect(reviewed).not_to_be_checked()
            reviewed.check()
            page.locator('#save').click()
            expect(page.locator('#notice')).to_contain_text('已保存到 metadata')
            assert fixture.store.records()['photo_1_1']['type']['placement'] == 'handheld'
            assert fixture.store.records()['photo_1_1']['reviewed']
            page.reload()
            expect(occlusion).to_have_value('foreground')
            expect(page.locator('.card[data-photo="photo_1_1"] [data-field="reviewed"]')).to_be_checked()
            card.locator('[data-field="placement"]').select_option('')
            expect(reviewed).to_be_disabled()
            expect(reviewed).not_to_be_checked()
            card.locator('[data-field="placement"]').select_option('handheld')
            card.locator('[data-field="background"]').select_option('multi_object')
            reviewed.check()

            # A competing edit must stop Save-and-next and preserve the unsaved draft.
            fixture.store.update('package_001', None, [dict(photo_id='photo_1_2',
                type=dict(placement='tabletop', background='multi_object'), reviewed=False, excluded=False)])
            page.locator('#save-next').click()
            expect(page.locator('#notice')).to_contain_text('其他页面')
            expect(page.locator('#package-info')).to_contain_text('package_001')
            expect(page.locator('#package-count')).to_contain_text('未保存修改')
            page.on('dialog', lambda dialog: dialog.accept())
            page.locator('#reload').click()
            expect(page.locator('.card[data-photo="photo_1_2"] [data-field="background"]')).to_have_value('multi_object')
            page.locator('.card[data-photo="photo_1_2"] [data-field="reviewed"]').check()
            page.locator('#save-next').click()
            expect(page.locator('#package-info')).to_contain_text('package_002')
            expect(page.locator('#review-text')).to_contain_text('2 / 3')
            page.locator('.card [data-field="reviewed"]').check()
            page.locator('#save-next').click()
            expect(page.locator('#notice')).to_contain_text('已全部校验并保存')
            expect(page.locator('#review-text')).to_contain_text('3 / 3')

            page.locator('#filter').select_option('pending')
            expect(page.locator('#packages .pkg')).to_have_count(0)
            page.locator('#filter').select_option('reviewed')
            expect(page.locator('#packages .pkg')).to_have_count(2)
            page.locator('#filter').select_option('occlusion_pending')
            expect(page.locator('#packages .pkg')).to_have_count(2)
            expect(page.locator('#next-pending')).to_be_enabled()
            page.reload()
            expect(page.locator('#filter')).to_have_value('occlusion_pending')
            expect(page.locator('#review-text')).to_contain_text('3 / 3')
            page.locator('#scope').select_option('all')
            expect(page.locator('#review-text')).to_contain_text('3 / 4')
            expect(page.locator('.card')).to_have_count(3)
            assert fixture.store.records()['photo_1_0'] == original
            card.locator('[data-action="exclude"]').click()
            page.locator('#save').click()
            expect(page.locator('#notice')).to_contain_text('已保存到 metadata')
            assert fixture.store.verify_active()['summary']['excluded'] == 1
            card.locator('[data-action="exclude"]').click()
            page.locator('#save').click()
            expect(page.locator('#notice')).to_contain_text('已保存到 metadata')
            assert fixture.store.verify_active()['summary']['excluded'] == 0
            # Batch occlusion annotations affect only selected photos and keep scene labels.
            second = page.locator('.card[data-photo="photo_1_2"]')
            card.locator('[data-field="selected"]').check()
            second.locator('[data-field="selected"]').check()
            page.locator('#batch-occlusion').select_option('none')
            page.locator('#batch-apply').click()
            expect(card.locator('[data-field="reviewed"]')).not_to_be_checked()
            expect(second.locator('[data-field="reviewed"]')).not_to_be_checked()
            card.locator('[data-field="reviewed"]').check()
            second.locator('[data-field="reviewed"]').check()
            page.locator('#save').click()
            expect(page.locator('#notice')).to_contain_text('已保存到 metadata')
            expect(page.locator('#occlusion-text')).to_contain_text('2 / 4')
            assert fixture.store.records()['photo_1_0'] == original
            assert fixture.store.records()['photo_1_1']['type']['background'] == 'single_object'
            page.locator('#filter').select_option('foreground_occlusion')
            expect(page.locator('#packages .pkg')).to_have_count(0)
            page.locator('#filter').select_option('all')
            assert (fixture.root / 'lists/photos.csv').read_bytes() == formal
            page.set_viewport_size({'width': 700, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            fixture.store.update('package_002', None, [dict(photo_id='photo_2_0',
                type=dict(placement='tabletop', background='single_object', occlusion='none'),
                reviewed=True, excluded=False)])
            page.goto(url + '/?scope=supplement&filter=occlusion_pending')
            expect(page.locator('#packages .pkg')).to_have_count(0)
            expect(page.locator('#occlusion-text')).to_contain_text('3 / 3')
            expect(page.locator('#notice')).to_contain_text('没有待处理照片')
            assert not errors, errors
            page.screenshot(path='/tmp/drugpack-manual-review.png')
            browser.close()
        print(json.dumps(dict(scope_from_receipt=True, explicit_confirmation=True,
            correction_and_reload=True, missing_labels_block_confirmation=True,
            stale_save_stops_navigation=True, saved_progress=True, pending_filter=True,
            original_outside_scope_preserved=True, exclusion_restore=True, formal_csv_unchanged=True,
            independent_occlusion_save_reload=True, batch_occlusion=True,
            occlusion_pending_navigation=True, completed_filter_progress=True)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        fixture.tearDown()


if __name__ == '__main__':
    main()
