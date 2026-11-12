import { describe, expect, it } from 'vitest'
import { SSEParser, type SSEMessage } from './sse'

function parseAll(chunks: string[]): SSEMessage[] {
  const parser = new SSEParser()
  return [...chunks.flatMap((chunk) => parser.feed(chunk)), ...parser.end()]
}

describe('SSEParser', () => {
  it('lit un événement complet avec type, id et données', () => {
    expect(parseAll(['event: cv_indexed\nid: 1\ndata: {"a":1}\n\n'])).toEqual([
      { event: 'cv_indexed', id: '1', data: '{"a":1}' },
    ])
  })

  it('reconstitue un événement coupé en plusieurs morceaux, même au milieu d’un champ', () => {
    const text = 'event: status\r\nid: 7\r\ndata: {"message":"reprise dans 60 s"}\r\n\r\n'
    for (const size of [1, 2, 3, 5, 8, 13]) {
      const chunks: string[] = []
      for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size))
      expect(parseAll(chunks)).toEqual([{ event: 'status', id: '7', data: '{"message":"reprise dans 60 s"}' }])
    }
  })

  it('gère un \\r\\n coupé entre deux morceaux sans créer de ligne vide en trop', () => {
    const parser = new SSEParser()
    expect(parser.feed('data: a\r')).toEqual([])
    expect(parser.feed('\ndata: b\r')).toEqual([])
    expect(parser.feed('\n\r\n')).toEqual([{ event: 'message', id: null, data: 'a\nb' }])
  })

  it('joint plusieurs lignes data par un saut de ligne', () => {
    expect(parseAll(['event: x\ndata: ligne 1\ndata: ligne 2\ndata:\ndata: ligne 4\n\n'])).toEqual([
      { event: 'x', id: null, data: 'ligne 1\nligne 2\n\nligne 4' },
    ])
  })

  it('ignore les commentaires et les pings, y compris entre deux lignes d’un événement', () => {
    const text = ': ping - 2026-11-10 10:00:00\r\n\r\nevent: a\r\n: commentaire\r\ndata: 1\r\n\r\n: ping\r\n\r\n'
    expect(parseAll([text])).toEqual([{ event: 'a', id: null, data: '1' }])
  })

  it('ignore les lignes vides en trop et les blocs sans data', () => {
    expect(parseAll(['\n\n\nevent: vide\n\n\nevent: b\ndata: 2\n\n\n'])).toEqual([
      { event: 'b', id: null, data: '2' },
    ])
  })

  it('accepte \\n, \\r\\n et \\r comme fins de ligne, et un BOM initial', () => {
    expect(parseAll(['﻿event: a\rdata: 1\r\rdata: 2\n\ndata: 3\r\n\r\n'])).toEqual([
      { event: 'a', id: null, data: '1' },
      { event: 'message', id: null, data: '2' },
      { event: 'message', id: null, data: '3' },
    ])
  })

  it('n’enlève qu’une seule espace après les deux-points', () => {
    expect(parseAll(['data:  deux espaces\ndata:sans espace\n\n'])[0]?.data).toBe(' deux espaces\nsans espace')
  })

  it('en fin de flux, n’émet pas un événement sans ligne vide finale', () => {
    expect(parseAll(['event: a\ndata: 1\n\nevent: b\ndata: incomplet'])).toEqual([
      { event: 'a', id: null, data: '1' },
    ])
    expect(parseAll(['event: a\ndata: 1\n'])).toEqual([])
  })

  it('en fin de flux, un \\r final termine sa ligne', () => {
    expect(parseAll(['data: 1\r\r'])).toEqual([{ event: 'message', id: null, data: '1' }])
  })

  it('garde le dernier id pour les événements suivants', () => {
    const messages = parseAll(['id: 4\ndata: a\n\ndata: b\n\n'])
    expect(messages.map((m) => m.id)).toEqual(['4', '4'])
  })
})
