// Parseur Server-Sent Events incrémental, écrit à la main (format de la spécification
// HTML « event stream ») : on lui donne les morceaux de texte dans l'ordre où ils
// arrivent, il renvoie les messages complets.

export interface SSEMessage {
  event: string
  data: string
  id: string | null
}

export class SSEParser {
  private buffer = ''
  private started = false
  private eventType = ''
  private dataLines: string[] = []
  private lastId: string | null = null

  /** Ajoute un morceau reçu ; renvoie les messages terminés par une ligne vide. */
  feed(chunk: string): SSEMessage[] {
    if (!this.started && chunk.length > 0) {
      this.started = true
      chunk = chunk.replace(/^﻿/, '') // BOM éventuel en tout début de flux
    }
    this.buffer += chunk
    const messages: SSEMessage[] = []
    let lineStart = 0
    for (let i = 0; i < this.buffer.length; i++) {
      const char = this.buffer[i]
      if (char !== '\n' && char !== '\r') continue
      if (char === '\r' && i + 1 === this.buffer.length) {
        // « \r » en fin de morceau : un « \n » peut suivre dans le prochain (« \r\n »).
        break
      }
      const line = this.buffer.slice(lineStart, i)
      if (char === '\r' && this.buffer[i + 1] === '\n') i++
      lineStart = i + 1
      this.processLine(line, messages)
    }
    this.buffer = this.buffer.slice(lineStart)
    return messages
  }

  /** Fin du flux. Un « \r » final termine sa ligne ; un message sans ligne vide finale est
   * incomplet et n'est pas émis (comme le prévoit la spécification). */
  end(): SSEMessage[] {
    const messages: SSEMessage[] = []
    if (this.buffer.endsWith('\r')) {
      this.processLine(this.buffer.slice(0, -1), messages)
    }
    this.buffer = ''
    this.eventType = ''
    this.dataLines = []
    return messages
  }

  private processLine(line: string, messages: SSEMessage[]): void {
    if (line === '') {
      this.dispatch(messages)
      return
    }
    if (line.startsWith(':')) return // commentaire (ping du serveur)
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    switch (field) {
      case 'event':
        this.eventType = value
        break
      case 'data':
        this.dataLines.push(value)
        break
      case 'id':
        if (!value.includes('\0')) this.lastId = value
        break
      default: // « retry » et champs inconnus : ignorés
        break
    }
  }

  private dispatch(messages: SSEMessage[]): void {
    if (this.dataLines.length > 0) {
      messages.push({
        event: this.eventType || 'message',
        data: this.dataLines.join('\n'),
        id: this.lastId,
      })
    }
    this.eventType = ''
    this.dataLines = []
  }
}
