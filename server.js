require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const axios = require('axios');
const vision = require('@google-cloud/vision');
let google = null;
try {
  ({ google } = require('googleapis'));
} catch (e) {
  console.error('Optional dependency "googleapis" is not available. Google Play subscription verification will be disabled until it is installed.', e?.message || e);
  google = null;
}

const app = express();
const PORT = process.env.PORT || 3000;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const VISION_API_URL = GOOGLE_API_KEY
  ? `https://vision.googleapis.com/v1/images:annotate?key=${GOOGLE_API_KEY}`
  : null;

const GOOGLE_APPLICATION_CREDENTIALS_JSON = process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON;
let visionClient = null;
if (GOOGLE_APPLICATION_CREDENTIALS_JSON) {
  try {
    let credsObj;
    try {
      credsObj = JSON.parse(GOOGLE_APPLICATION_CREDENTIALS_JSON);
    } catch (_) {
      const decoded = Buffer.from(GOOGLE_APPLICATION_CREDENTIALS_JSON, 'base64').toString('utf8');
      credsObj = JSON.parse(decoded);
    }

    visionClient = new vision.ImageAnnotatorClient({
      credentials: credsObj
    });
  } catch (e) {
    console.error('Failed to initialize Google Vision client from GOOGLE_APPLICATION_CREDENTIALS_JSON:', e?.message || e);
    visionClient = null;
  }
}

const ANDROID_PACKAGE_NAME = process.env.ANDROID_PACKAGE_NAME || 'com.hangly.hanglyapp';
const GOOGLE_PLAY_SERVICE_ACCOUNT_JSON = process.env.GOOGLE_PLAY_SERVICE_ACCOUNT_JSON;
let playPublisherClient = null;

const ENFORCE_PREMIUM = (process.env.ENFORCE_PREMIUM || '').toString().trim().toLowerCase() === 'true';

function initPlayPublisherClient() {
  if (!GOOGLE_PLAY_SERVICE_ACCOUNT_JSON) return null;
  if (!google) {
    console.error('googleapis is not installed, cannot initialize Google Play Publisher client.');
    return null;
  }
  try {
    let credsObj;
    try {
      credsObj = JSON.parse(GOOGLE_PLAY_SERVICE_ACCOUNT_JSON);
    } catch (_) {
      const decoded = Buffer.from(GOOGLE_PLAY_SERVICE_ACCOUNT_JSON, 'base64').toString('utf8');
      credsObj = JSON.parse(decoded);
    }

    const auth = new google.auth.GoogleAuth({
      credentials: credsObj,
      scopes: ['https://www.googleapis.com/auth/androidpublisher']
    });

    return google.androidpublisher({ version: 'v3', auth });
  } catch (e) {
    console.error('Failed to initialize Google Play Publisher client:', e?.message || e);
    return null;
  }
}

playPublisherClient = initPlayPublisherClient();

function parsePremiumHeaders(req) {
  const productId = (req.headers['x-play-product-id'] || '').toString().trim();
  const purchaseToken = (req.headers['x-play-purchase-token'] || '').toString().trim();
  return { productId, purchaseToken };
}

const premiumCache = new Map();
const PREMIUM_CACHE_TTL_MS = 2 * 60 * 1000;

async function verifyGooglePlaySubscription({ packageName, productId, purchaseToken }) {
  if (!playPublisherClient) {
    return { isPremium: false, reason: 'not_configured' };
  }
  if (!packageName || !productId || !purchaseToken) {
    return { isPremium: false, reason: 'missing_token' };
  }

  const cacheKey = `${packageName}:${productId}:${purchaseToken}`;
  const cached = premiumCache.get(cacheKey);
  if (cached && (Date.now() - cached.ts) < PREMIUM_CACHE_TTL_MS) {
    return cached.value;
  }

  try {
    const resp = await playPublisherClient.purchases.subscriptions.get({
      packageName,
      subscriptionId: productId,
      token: purchaseToken
    });

    const data = resp?.data || {};
    const expiryTimeMillis = Number(data.expiryTimeMillis);
    const nowMs = Date.now();
    const isActive = Number.isFinite(expiryTimeMillis) && expiryTimeMillis > nowMs;

    const value = {
      isPremium: Boolean(isActive),
      expiryTimeMillis: Number.isFinite(expiryTimeMillis) ? expiryTimeMillis : null,
      raw: {
        paymentState: data.paymentState,
        cancelReason: data.cancelReason,
        acknowledgementState: data.acknowledgementState
      }
    };

    premiumCache.set(cacheKey, { ts: Date.now(), value });
    return value;
  } catch (e) {
    const status = e?.response?.status;
    const msg = e?.response?.data?.error?.message || e?.message;
    console.error('Google Play subscription verification failed:', {
      status,
      message: msg
    });
    return { isPremium: false, reason: 'verify_failed', status, message: msg };
  }
}

async function requirePremium(req, res, next) {
  try {
    if (!ENFORCE_PREMIUM) {
      req.premium = { isPremium: false, skipped: true, reason: 'enforcement_disabled' };
      return next();
    }

    const { productId, purchaseToken } = parsePremiumHeaders(req);
    const result = await verifyGooglePlaySubscription({
      packageName: ANDROID_PACKAGE_NAME,
      productId,
      purchaseToken
    });

    if (result?.reason === 'not_configured') {
      req.premium = { isPremium: false, skipped: true, reason: 'not_configured' };
      return next();
    }

    if (!result.isPremium) {
      return res.status(402).json({
        success: false,
        error: 'premium_required',
        message: 'Premium subscription required',
        details: result
      });
    }

    req.premium = result;
    return next();
  } catch (e) {
    return res.status(500).json({
      success: false,
      error: 'premium_check_failed',
      message: e?.message || 'Premium check failed'
    });
  }
}

function detectIndianChatLanguage(text) {
  const t = (text || '').toString();
  if (!t.trim()) return 'hinglish';

  const devanagariMatches = t.match(/[\u0900-\u097F]/g) || [];
  const latinMatches = t.match(/[A-Za-z]/g) || [];
  const totalLetters = devanagariMatches.length + latinMatches.length;

  if (totalLetters === 0) return 'hinglish';

  const devanagariRatio = devanagariMatches.length / totalLetters;
  if (devanagariRatio > 0.55) return 'hindi';

  const hinglishSignals = [
    /\b(yaar|yr|bro|bhai|behen|bhen|didi|babu|jaan|ya|kyu|kyun|kya|kaise|nahi|haan|haa|ha|matlab|scene|vibe|chal|arey|arrey|hoga|nhi|nai)\b/i,
    /\b(lol|lmao|bruh|vibe|scene|sahi|mast|pagal|crazy)\b/i
  ];
  const hasHinglishSignal = hinglishSignals.some((re) => re.test(t));
  if (hasHinglishSignal) return 'hinglish';

  return 'english';
}

function pickReplyLengthHint(spicinessLevel) {
  const lvl = Number.isFinite(Number(spicinessLevel)) ? Number(spicinessLevel) : 1;
  const clamped = Math.min(Math.max(lvl, 1), 3);
  if (clamped === 1) return '1 short line (max ~12 words).';
  if (clamped === 2) return '1-2 short lines (max ~20 words).';
  return '1-2 short lines (max ~22 words).';
}

function spiceStyleGuide(spicinessLevel) {
  const lvl = Number.isFinite(Number(spicinessLevel)) ? Number(spicinessLevel) : 1;
  const clamped = Math.min(Math.max(lvl, 1), 3);
  return {
    1: 'sweet, confident, low-key charming. Safe and not over-flirty.',
    2: 'playful banter, light tease, clear interest. Still classy.',
    3: 'high-rizz, bold flirt, witty. Spicy but not vulgar, no explicit sexual content.'
  }[clamped];
}

function extractRecentChatText(ocrText, { maxLines = 28, maxChars = 1600 } = {}) {
  const raw = (ocrText || '').toString();
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length === 0) return '';

  const recent = lines.slice(Math.max(0, lines.length - maxLines));
  const joined = recent.join('\n');
  if (joined.length <= maxChars) return joined;
  return joined.slice(joined.length - maxChars);
}

function buildReplyRizzlerPrompt({ ocrText, userContext, spicinessLevel }) {
  const ocr = (ocrText || '').toString().trim();
  const ocrRecent = extractRecentChatText(ocr);
  const ctx = (userContext || '').toString().trim();
  const combined = [ocr, ctx].filter(Boolean).join('\n');
  const lang = detectIndianChatLanguage(combined);
  const lengthHint = pickReplyLengthHint(spicinessLevel);
  const spiceGuide = spiceStyleGuide(spicinessLevel);

  const languageRule =
    lang === 'hindi'
      ? 'Write in Hindi (Devanagari).'
      : lang === 'hinglish'
        ? 'Write in Hinglish (Hindi in Latin letters + a bit of English, like Indian Gen Z texting).'
        : 'Write in English, but keep it India-friendly (natural texting).';

  return {
    lang,
    messages: [
      {
        role: 'system',
        content: [
          'You are Reply Rizzler — you generate ONE reply message for Indian Gen Z texting.',
          'Never mention AI, OCR, analysis, or policy.',
          'No cringe, no try-hard pickup lines, no overuse of emojis.',
          'No slurs. No explicit sexual content. Keep it respectful.',
          'CRITICAL: Your reply must be SPECIFIC to the chat. Reference at least one concrete detail (a plan, a time, a place, a joke, a name, a question) from the last messages if available.',
          'CRITICAL: Reply to the LAST INCOMING message in the chat (the most recent thing the other person said). Do not respond to older parts of the conversation.',
          'Do NOT add quotes. Output ONLY the final reply text.'
        ].join('\n')
      },
      {
        role: 'user',
        content: [
          `Goal: craft ONE reply to the last message in this chat screenshot text.`,
          `Language: ${languageRule}`,
          `Vibe: ${spiceGuide}`,
          `Length: ${lengthHint}`,
          'Important: Match the chat’s vibe (formal/informal), and mirror any slang level naturally.',
          'If the last incoming message contains a question, answer it directly. If it proposes a plan, confirm/deny with a clear next step.',
          ctx ? `Extra context from user (may be important):\n${ctx}` : null,
          `Chat text (OCR - MOST RECENT LINES ONLY):\n${ocrRecent || '(empty)'}`
        ].filter(Boolean).join('\n\n')
      }
    ]
  };
}

function buildGlowUpPrompt({ message, tone, includeEmojis }) {
  const msg = (message || '').toString().trim();
  const toneStr = (tone || '').toString().trim();
  const lang = detectIndianChatLanguage(msg);

  const languageRule =
    lang === 'hindi'
      ? 'MANDATORY: Write in Hindi using Devanagari script ONLY. If the user wrote in Hindi, you MUST reply in Hindi.'
      : lang === 'hinglish'
        ? 'MANDATORY: Write in Hinglish ONLY (Hindi words written in Latin/English letters like "kya", "nahi", "mast"). Mix with some English words. This is CRITICAL: Hinglish input must produce Hinglish output, never pure English.'
        : 'Write in English (India-friendly Gen Z texting style).';

  const emojiRule = includeEmojis
    ? 'CRITICAL EMOJI RULE: Add 1-3 emojis. Place them at RANDOM positions - some at the start, some in the middle of sentences, some at the end. Do NOT put all emojis only at the end. Examples: "✨ finally done" OR "that was amazing 🔥 yesterday" OR "let\'s go 🚀 meet at 5"'
    : 'Do not add any emojis.';

  const toneRules = {
    'flirty': 'Tone: Flirty and playful. Show romantic interest with confidence. Use compliments, teasing, and charm. Be bold but respectful.',
    'funny': 'Tone: Funny and witty. Make them laugh with humor, puns, or playful sarcasm. Keep it light and entertaining.',
    'savage': 'Tone: Savage and bold. Confident, slightly cocky, with witty comebacks. Edgy but not rude.',
    'confident': 'Tone: Confident and direct. No hesitation, no over-explaining. Short, punchy, and self-assured.',
    'friendly': 'Tone: Warm and friendly. Casual, approachable, like talking to a close friend.',
    'smooth': 'Tone: Smooth and charming. Sophisticated, effortless cool. James Bond vibes but casual.',
    'casual': 'Tone: Casual and laid-back. Chill, easy-going, no pressure.'
  };

  const toneRule = toneRules[toneStr?.toLowerCase()] || `Tone: ${toneStr} (adapt naturally for Indian Gen Z texting).`;

  return {
    lang,
    messages: [
      {
        role: 'system',
        content: [
          'You rewrite short messages to sound better for texting in India (WhatsApp/Insta/dating apps).',
          'Never mention AI, instructions, or analysis.',
          'Keep the SAME meaning - do not add new facts or change the core message.',
          'No cringe pickup lines, no over-explaining. Sound like a real human texting.',
          'You MUST generate 3 variations that are MEANINGFULLY DIFFERENT.',
          'Variation A: concise + confident (short, direct).',
          'Variation B: playful + warm (slightly more expressive).',
          'Variation C: bold + witty (more attitude, still respectful).',
          'CRITICAL: All three must have different openings and different sentence structure. No near-duplicates.',
          'Output MUST be valid JSON only, with this exact shape: {"variants":["...","...","..."]}.',
          'No markdown. No extra keys. No quotes around the whole JSON string.'
        ].join('\n')
      },
      {
        role: 'user',
        content: [
          `Task: Rewrite this message to make it hit better.`,
          ``,
          `INPUT LANGUAGE (MUST MATCH): ${languageRule}`,
          ``,
          `TONE (apply this style): ${toneRule}`,
          ``,
          `${emojiRule}`,
          ``,
          `IMPORTANT: Make it smooth, natural, and authentic. Each variant should be 1-2 short lines max.`,
          ``,
          `Original message:\n${msg}`
        ].join('\n')
      }
    ]
  };
}

function parseGlowUpVariants(raw, requestedCount) {
  const n = Math.max(1, Math.min(3, Math.floor(Number(requestedCount) || 1)));
  const text = (raw || '').toString().trim();
  if (!text) return [];

  const tryJson = () => {
    try {
      const start = text.indexOf('{');
      const end = text.lastIndexOf('}');
      if (start === -1 || end === -1 || end <= start) return null;
      const jsonStr = text.slice(start, end + 1);
      const obj = JSON.parse(jsonStr);
      if (!obj || !Array.isArray(obj.variants)) return null;
      const items = obj.variants
        .map((s) => sanitizeSingleTextReply((s || '').toString()))
        .filter(Boolean);
      return items;
    } catch (_) {
      return null;
    }
  };

  const jsonItems = tryJson();
  if (jsonItems && jsonItems.length) return jsonItems.slice(0, n);

  // Fallback: split lines and sanitize
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => l.replace(/^[-*\d.]+\s*/, ''))
    .map((l) => sanitizeSingleTextReply(l))
    .filter(Boolean);

  return lines.slice(0, n);
}

function sanitizeSingleTextReply(raw) {
  if (!raw || typeof raw !== 'string') return '';

  let s = raw.trim();

  // Remove code fences if the model accidentally returns them
  s = s.replace(/^```[a-zA-Z]*\s*/g, '').replace(/```\s*$/g, '').trim();

  // Remove surrounding quotes
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith('\'') && s.endsWith('\''))) {
    s = s.slice(1, -1).trim();
  }

  // If multiple lines are returned, keep the first meaningful 1-2 lines
  const lines = s
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length === 0) return '';
  if (lines.length === 1) return lines[0];

  // Keep up to 2 lines max (still feels like a single chat message)
  return `${lines[0]}\n${lines[1]}`.trim();
}

console.log('Vision auth mode:', visionClient ? 'service_account' : (VISION_API_URL ? 'api_key' : 'not_configured'));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const OPENAI_MODEL_FALLBACKS = (process.env.OPENAI_MODEL_FALLBACKS || 'gpt-4o-mini')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

async function callOpenAIChatCompletionsWithFallback(payload, { preferredModel }) {
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not configured');
  }

  const modelsToTry = [preferredModel, ...OPENAI_MODEL_FALLBACKS]
    .filter(Boolean)
    .filter((m, i, arr) => arr.indexOf(m) === i);

  let lastError = null;
  for (const model of modelsToTry) {
    try {
      const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        { ...payload, model },
        {
          headers: {
            Authorization: `Bearer ${OPENAI_API_KEY}`,
            'Content-Type': 'application/json'
          },
          timeout: 25000
        }
      );
      return response;
    } catch (err) {
      lastError = err;
      const status = err?.response?.status;
      const msg = err?.response?.data?.error?.message || err?.message || '';
      const isModelAccessDenied = status === 403 && /does not have access to model/i.test(msg);
      if (isModelAccessDenied) {
        continue;
      }
      throw err;
    }
  }

  throw lastError || new Error('OpenAI request failed');
}

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file uploads (in-memory only; do not persist user images)
const storage = multer.memoryStorage();

// File filter function
const fileFilter = function(req, file, cb) {
  const filetypes = /jpeg|jpg|png|gif/;
  const mimetype = filetypes.test(file.mimetype);
  const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
  
  if (mimetype && extname) {
    return cb(null, true);
  }
  cb(new Error('Only image files are allowed (JPEG, JPG, PNG, GIF)'));
};

const upload = multer({ 
  storage: storage,
  limits: { 
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 1
  },
  fileFilter: fileFilter
});

// Function to analyze image using Google Cloud Vision API
async function analyzeImageWithVision(imageBuffer) {
  try {
    const base64Image = imageBuffer.toString('base64');

    if (visionClient) {
      const [result] = await visionClient.annotateImage({
        image: { content: base64Image },
        features: [{ type: 'TEXT_DETECTION', maxResults: 10 }]
      });
      return { responses: [result] };
    }

    if (!VISION_API_URL) {
      throw new Error('Google Vision credentials are not configured (set GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_API_KEY)');
    }

    const request = {
      requests: [
        {
          image: {
            content: base64Image
          },
          features: [
            {
              type: 'TEXT_DETECTION',
              maxResults: 10
            }
          ]
        }
      ]
    };

    const response = await axios.post(VISION_API_URL, request, {
      timeout: 25000 // 25 second timeout
    });
    return response.data;
  } catch (error) {
    const status = error?.response?.status;
    const data = error?.response?.data;
    console.error('Error analyzing image with Vision API:', {
      message: error?.message,
      status,
      data
    });
    throw error;
  }
}

async function generateOpenAIResponse({ text, spicinessLevel, context }) {
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not configured');
  }

  const userContext = (context || '').toString().trim();
  const ocrText = (text || '').toString().trim();
  const prompt = buildReplyRizzlerPrompt({
    ocrText,
    userContext,
    spicinessLevel
  });

  const payload = {
    messages: prompt.messages,
    temperature: 0.75,
    presence_penalty: 0.2,
    max_tokens: 160
  };

  const response = await callOpenAIChatCompletionsWithFallback(payload, { preferredModel: OPENAI_MODEL });

  const content = response?.data?.choices?.[0]?.message?.content;
  if (!content || typeof content !== 'string') {
    throw new Error('No response from OpenAI');
  }
  const cleaned = sanitizeSingleTextReply(content);
  if (!cleaned) {
    throw new Error('Empty response from OpenAI');
  }
  return cleaned;
}

// Routes
app.get('/', (req, res) => {
  res.send('Hangly Backend is running!');
});

app.post('/api/glowup', requirePremium, async (req, res) => {
  try {
    const { message, tone, include_emojis, count } = req.body || {};

    const msg = (message || '').toString().trim();
    const toneStr = (tone || '').toString().trim();
    const includeEmojis = include_emojis === true || include_emojis === 'true' || include_emojis === 1 || include_emojis === '1';
    const requestedCountRaw = Number(count);
    const requestedCount = Number.isFinite(requestedCountRaw) ? requestedCountRaw : 1;
    const n = Math.max(1, Math.min(3, Math.floor(requestedCount)));

    if (!msg) {
      return res.status(400).json({
        success: false,
        error: 'Message is required',
        message: 'Message is required'
      });
    }

    if (!OPENAI_API_KEY) {
      return res.status(500).json({
        success: false,
        error: 'OPENAI_API_KEY is not configured',
        message: 'OPENAI_API_KEY is not configured'
      });
    }

    const glowPrompt = buildGlowUpPrompt({
      message: msg,
      tone: toneStr,
      includeEmojis
    });

    const payload = {
      messages: glowPrompt.messages,
      temperature: 0.95,
      presence_penalty: 0.6,
      frequency_penalty: 0.35,
      max_tokens: 200,
      n: 1
    };

    const response = await callOpenAIChatCompletionsWithFallback(payload, { preferredModel: OPENAI_MODEL });

    const choices = response?.data?.choices;
    if (!Array.isArray(choices) || choices.length === 0) {
      throw new Error('No response from OpenAI');
    }

    const content = choices?.[0]?.message?.content;
    const improvedMessages = parseGlowUpVariants(content, n);

    if (improvedMessages.length === 0) {
      throw new Error('Empty response from OpenAI');
    }

    return res.json({
      success: true,
      improved_message: improvedMessages[0],
      improved_messages: improvedMessages.slice(0, n),
      tone: toneStr,
      include_emojis: includeEmojis,
      count: n
    });
  } catch (error) {
    const upstreamStatus = error?.response?.status;
    const upstreamData = error?.response?.data;
    const upstreamMessage =
      upstreamData?.error?.message ||
      upstreamData?.error?.type ||
      upstreamData?.error?.code ||
      (typeof upstreamData === 'string' ? upstreamData : null);

    console.error('Unexpected error in /api/glowup:', {
      message: error?.message,
      upstream_status: upstreamStatus,
      upstream_message: upstreamMessage,
      upstream_data: upstreamData
    });
    return res.status(500).json({
      success: false,
      error: 'Failed to glow up message',
      message: upstreamMessage || error?.message || 'Failed to glow up message',
      details: upstreamData || error?.message,
      upstream_status: upstreamStatus
    });
  }
});

// Analyze image and generate response
app.post('/api/analyze', requirePremium, (req, res) => {
  upload.single('image')(req, res, async (err) => {
    try {
      if (err instanceof multer.MulterError) {
        console.error('Multer error:', err);
        return res.status(400).json({ 
          error: 'File upload error',
          details: err.message 
        });
      } else if (err) {
        console.error('Upload error:', err);
        return res.status(400).json({ 
          error: 'File upload failed',
          details: err.message 
        });
      }

      const { spiciness_level, context } = req.body;
      const imageBuffer = req.file?.buffer;

      console.log('Received /api/analyze request', {
        has_file: Boolean(req.file),
        mimetype: req.file?.mimetype,
        size: req.file?.size,
        spiciness_level
      });

      if (!imageBuffer) {
        return res.status(400).json({ 
          error: 'No image file provided',
          receivedFiles: req.files,
          body: req.body
        });
      }

      try {
        // Analyze image with Google Cloud Vision
        console.log('Analyzing image with Vision API...');
        const visionResult = await analyzeImageWithVision(imageBuffer);
        
        // Extract labels from vision result
        const labels = visionResult.responses[0]?.labelAnnotations?.map(label => label.description) || [];
        const text = visionResult.responses[0]?.textAnnotations?.[0]?.description || '';

        console.log('Vision extracted text length:', (text || '').length);
        
        // Generate response based on detected content
        const rawLevel = parseInt(spiciness_level || '1');
        const spicinessInt = Number.isFinite(rawLevel) ? Math.min(Math.max(rawLevel, 1), 3) : 1;
        let response;
        if (!text || text.trim().length === 0) {
          throw new Error('No readable chat text detected in the image. Try a clearer screenshot (crop, increase brightness, avoid blur).');
        }

        try {
          response = await generateOpenAIResponse({
            text,
            spicinessLevel: spicinessInt,
            context
          });
        } catch (openAiError) {
          console.error('OpenAI error in /api/analyze:', {
            message: openAiError?.message,
            upstream_status: openAiError?.response?.status,
            upstream_data: openAiError?.response?.data
          });
          throw new Error('AI is temporarily unavailable. Please try again in a few seconds.');
        }
        console.log('Generated response:', response);

        // Send the response
        res.json({
          success: true,
          response: response,
          spiciness_level: spicinessInt,
          detected_labels: labels,
          detected_text: text
        });

      } catch (processError) {
        console.error('Error processing image:', processError);

        const upstreamStatus = processError?.response?.status;
        const upstreamData = processError?.response?.data;
        const upstreamMessage =
          upstreamData?.error?.message ||
          upstreamData?.error?.status ||
          (typeof upstreamData === 'string' ? upstreamData : null);
        
        const statusCode = /No readable chat text detected/i.test(processError?.message || '') ? 400 : 500;
        return res.status(statusCode).json({ 
          error: 'Failed to process image',
          message: upstreamMessage || processError.message,
          details: upstreamData || processError.message,
          upstream_status: upstreamStatus
        });
      }
    } catch (error) {
      console.error('Unexpected error in /api/analyze:', error);
      res.status(500).json({ 
        error: 'Internal server error',
        message: error.message,
        details: error.message
      });
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Something went wrong!',
    message: err.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
