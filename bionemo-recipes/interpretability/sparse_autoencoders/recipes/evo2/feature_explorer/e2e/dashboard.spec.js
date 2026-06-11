// Smoke test: the dashboard renders and each live tab round-trips against the mock backend.
// Selectors are derived from the component JSX (button labels / placeholders / summary text).
import { test, expect } from '@playwright/test'

const TABS = ['Feature atlas', 'Generative steering', 'Sequence inspector', 'Sequence UMAP']

test('all four tabs render', async ({ page }) => {
  await page.goto('/')
  for (const t of TABS) {
    await expect(page.getByRole('button', { name: t })).toBeVisible()
  }
})

test('Sequence inspector annotates a pasted sequence', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Sequence inspector' }).click()
  await page.getByPlaceholder(/Paste DNA/i).fill('ACGTACGTACGTACGTACGT')
  const go = page.getByRole('button', { name: /Annotate/i })
  await expect(go).toBeEnabled()
  await go.click()
  await expect(page.getByText(/motif_ATG/).first()).toBeVisible() // a labeled feature from the mock
})

test('Generative steering generates a sequence', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Generative steering' }).click()
  const go = page.getByRole('button', { name: /Generate/i })
  await expect(go).toBeEnabled()
  await go.click()
  await expect(page.getByText(/generation/i).first()).toBeVisible() // the "Generation" result block
})

test('Sequence UMAP embeds pasted sequences', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Sequence UMAP' }).click()
  await page.getByRole('button', { name: /Paste your own/i }).click()
  await page.getByPlaceholder(/MYSEQ/i).fill('>s1|a\nACGTACGTACGT\n>s2|b\nTTGGCCAATTGG\n>s3|c\nGGGGCCCCAAAA')
  const go = page.getByRole('button', { name: /Embed/i })
  await expect(go).toBeEnabled()
  await go.click()
  await expect(page.getByText(/sequences ×/).first()).toBeVisible() // "N sequences × M features" summary
})
