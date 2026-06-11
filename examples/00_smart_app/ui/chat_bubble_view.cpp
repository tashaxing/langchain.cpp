// chat_bubble_view.cpp -- WeChat-style chat bubble view with Markdown rendering.
#include "chat_bubble_view.h"

#include <FL/Fl.H>
#include <FL/fl_draw.H>
#include <md4c.h>

#include <algorithm>
#include <sstream>

namespace smart_app::ui
{

namespace
{

constexpr int kPadding = 14;
constexpr int kGap = 12;
constexpr int kAvatar = 32;
constexpr int kBubblePadX = 12;
constexpr int kBubblePadY = 9;
constexpr int kTextSize = 14;
constexpr int kLineHeight = 20;

enum class BlockKind
{
    Paragraph,
    Heading,
    ListItem,
    Quote,
    CodeBlock,
    Rule,
    Table,
    TableRow
};

struct SpanStyle
{
    bool strong = false;
    bool em = false;
    bool code = false;
    bool link = false;
    bool strike = false;
    bool underline = false;
    bool image = false;
    bool math = false;
};

bool same_style(const SpanStyle& a, const SpanStyle& b)
{
    return a.strong == b.strong && a.em == b.em && a.code == b.code &&
           a.link == b.link && a.strike == b.strike &&
           a.underline == b.underline && a.image == b.image && a.math == b.math;
}

struct MarkdownSpan
{
    std::string text;
    SpanStyle style;
};

struct TableCell
{
    std::vector<MarkdownSpan> spans;
    bool header = false;
    MD_ALIGN align = MD_ALIGN_DEFAULT;
};

struct TableRow
{
    std::vector<TableCell> cells;
};

struct MarkdownBlock
{
    BlockKind kind = BlockKind::Paragraph;
    std::vector<MarkdownSpan> spans;
    std::vector<TableRow> table_rows;
    int level = 0;
    bool ordered = false;
    int list_number = 0;
    bool task = false;
    bool task_checked = false;
};

struct ParseContext
{
    std::vector<MarkdownBlock> blocks;
    MarkdownBlock current;
    MarkdownBlock table;
    TableRow current_row;
    TableCell current_cell;
    SpanStyle style;
    bool collecting = false;
    bool in_table = false;
    bool in_table_row = false;
    bool in_table_cell = false;
    bool current_cell_header = false;
    std::vector<bool> ordered_stack;
    std::vector<int> ordered_counter;
    int quote_depth = 0;
    int list_depth = 0;
};

struct RenderSpan
{
    std::string text;
    SpanStyle style;
    int width = 0;
};

struct RenderLine
{
    std::vector<RenderSpan> spans;
    int width = 0;
};

int span_font(const SpanStyle& style, const MarkdownBlock& block)
{
    if (style.code || block.kind == BlockKind::CodeBlock)
    {
        return style.strong ? FL_COURIER_BOLD : FL_COURIER;
    }
    if (block.kind == BlockKind::Heading || style.strong)
    {
        return style.em ? FL_HELVETICA_BOLD_ITALIC : FL_HELVETICA_BOLD;
    }
    if (style.em)
    {
        return FL_HELVETICA_ITALIC;
    }
    return FL_HELVETICA;
}

int block_size(const MarkdownBlock& block)
{
    if (block.kind == BlockKind::Heading)
    {
        return block.level <= 1 ? 20 : (block.level == 2 ? 18 : 16);
    }
    return kTextSize;
}

int block_line_height(const MarkdownBlock& block)
{
    return block.kind == BlockKind::Heading ? block_size(block) + 8 : kLineHeight;
}

std::size_t utf8_char_len(const std::string& s, std::size_t i)
{
    if (i >= s.size()) return 0;
    unsigned char c = static_cast<unsigned char>(s[i]);
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return i + 2 <= s.size() ? 2 : 1;
    if ((c & 0xF0) == 0xE0) return i + 3 <= s.size() ? 3 : 1;
    if ((c & 0xF8) == 0xF0) return i + 4 <= s.size() ? 4 : 1;
    return 1;
}

int measure_text(const std::string& text, int font, int size)
{
    fl_font(font, size);
    int w = 0, h = 0;
    fl_measure(text.c_str(), w, h, 0);
    return w;
}

void add_render_piece(RenderLine& line, const std::string& text, const SpanStyle& style, int width)
{
    if (!line.spans.empty() && same_style(line.spans.back().style, style))
    {
        line.spans.back().text += text;
        line.spans.back().width += width;
    }
    else
    {
        line.spans.push_back({text, style, width});
    }
    line.width += width;
}

std::vector<RenderLine> layout_spans(const MarkdownBlock& block, int max_width)
{
    std::vector<RenderLine> lines;
    RenderLine line;
    int size = block_size(block);

    auto flush_line = [&]() {
        lines.push_back(std::move(line));
        line = RenderLine{};
    };

    for (const auto& span : block.spans)
    {
        int font = span_font(span.style, block);
        for (std::size_t i = 0; i < span.text.size();)
        {
            if (span.text[i] == '\n')
            {
                flush_line();
                ++i;
                continue;
            }

            std::size_t len = utf8_char_len(span.text, i);
            std::string piece = span.text.substr(i, len);
            int piece_w = measure_text(piece, font, size);
            if (line.width > 0 && line.width + piece_w > max_width)
            {
                flush_line();
            }
            add_render_piece(line, piece, span.style, piece_w);
            i += len;
        }
    }

    if (!line.spans.empty() || lines.empty())
    {
        lines.push_back(std::move(line));
    }
    return lines;
}

void append_text(ParseContext& ctx, std::string text, SpanStyle style)
{
    if (ctx.in_table_cell)
    {
        if (!ctx.current_cell.spans.empty() && same_style(ctx.current_cell.spans.back().style, style))
        {
            ctx.current_cell.spans.back().text += std::move(text);
        }
        else
        {
            ctx.current_cell.spans.push_back({std::move(text), style});
        }
        return;
    }

    if (!ctx.collecting)
    {
        return;
    }
    if (!ctx.current.spans.empty() && same_style(ctx.current.spans.back().style, style))
    {
        ctx.current.spans.back().text += std::move(text);
    }
    else
    {
        ctx.current.spans.push_back({std::move(text), style});
    }
}

bool block_empty(const MarkdownBlock& block)
{
    for (const auto& span : block.spans)
    {
        if (!span.text.empty()) return false;
    }
    return true;
}

void trim_trailing_newline(MarkdownBlock& block)
{
    while (!block.spans.empty())
    {
        auto& text = block.spans.back().text;
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
        {
            text.pop_back();
        }
        if (!text.empty()) break;
        block.spans.pop_back();
    }
}

void finish_current(ParseContext& ctx)
{
    if (!ctx.collecting)
    {
        return;
    }
    trim_trailing_newline(ctx.current);
    if (!block_empty(ctx.current) || ctx.current.kind == BlockKind::Rule)
    {
        if (ctx.quote_depth > 0 && ctx.current.kind == BlockKind::Paragraph)
        {
            ctx.current.kind = BlockKind::Quote;
        }
        ctx.blocks.push_back(ctx.current);
    }
    ctx.current = MarkdownBlock{};
    ctx.collecting = false;
}

int enter_block(MD_BLOCKTYPE type, void* detail, void* userdata)
{
    auto* ctx = static_cast<ParseContext*>(userdata);
    switch (type)
    {
    case MD_BLOCK_QUOTE:
        ++ctx->quote_depth;
        break;
    case MD_BLOCK_UL:
        ++ctx->list_depth;
        ctx->ordered_stack.push_back(false);
        ctx->ordered_counter.push_back(0);
        break;
    case MD_BLOCK_OL:
        ++ctx->list_depth;
        ctx->ordered_stack.push_back(true);
        ctx->ordered_counter.push_back(0);
        break;
    case MD_BLOCK_TABLE:
        finish_current(*ctx);
        ctx->table = MarkdownBlock{};
        ctx->table.kind = BlockKind::Table;
        ctx->in_table = true;
        break;
    case MD_BLOCK_TR:
        if (ctx->in_table)
        {
            ctx->current_row = TableRow{};
            ctx->in_table_row = true;
        }
        break;
    case MD_BLOCK_TH:
    case MD_BLOCK_TD:
        if (ctx->in_table_row)
        {
            ctx->current_cell = TableCell{};
            ctx->current_cell.header = type == MD_BLOCK_TH;
            if (detail)
            {
                ctx->current_cell.align = static_cast<MD_BLOCK_TD_DETAIL*>(detail)->align;
            }
            ctx->in_table_cell = true;
        }
        break;
    case MD_BLOCK_P:
        if (ctx->in_table_cell)
        {
            break;
        }
        finish_current(*ctx);
        ctx->current = {ctx->quote_depth > 0 ? BlockKind::Quote : BlockKind::Paragraph, {}, {}, 0};
        ctx->collecting = true;
        break;
    case MD_BLOCK_H:
        finish_current(*ctx);
        ctx->current = {BlockKind::Heading, {}, {}, 1};
        if (detail)
        {
            ctx->current.level = static_cast<MD_BLOCK_H_DETAIL*>(detail)->level;
        }
        ctx->collecting = true;
        break;
    case MD_BLOCK_CODE:
        finish_current(*ctx);
        ctx->current = {BlockKind::CodeBlock, {}, {}, 0};
        ctx->collecting = true;
        break;
    case MD_BLOCK_LI:
        finish_current(*ctx);
        ctx->current = {BlockKind::ListItem, {}, {}, ctx->list_depth};
        if (!ctx->ordered_stack.empty())
        {
            ctx->current.ordered = ctx->ordered_stack.back();
            if (ctx->current.ordered)
            {
                ctx->current.list_number = ++ctx->ordered_counter.back();
            }
        }
        if (detail)
        {
            auto* li = static_cast<MD_BLOCK_LI_DETAIL*>(detail);
            ctx->current.task = li->is_task != 0;
            ctx->current.task_checked = li->is_task && (li->task_mark == 'x' || li->task_mark == 'X');
        }
        ctx->collecting = true;
        break;
    case MD_BLOCK_HR:
        finish_current(*ctx);
        ctx->blocks.push_back({BlockKind::Rule, {}, {}, 0});
        break;
    default:
        break;
    }
    return 0;
}

int leave_block(MD_BLOCKTYPE type, void*, void* userdata)
{
    auto* ctx = static_cast<ParseContext*>(userdata);
    switch (type)
    {
    case MD_BLOCK_P:
        if (!ctx->in_table_cell)
        {
            finish_current(*ctx);
        }
        break;
    case MD_BLOCK_H:
    case MD_BLOCK_CODE:
    case MD_BLOCK_LI:
        finish_current(*ctx);
        break;
    case MD_BLOCK_TH:
    case MD_BLOCK_TD:
        if (ctx->in_table_cell)
        {
            ctx->current_row.cells.push_back(ctx->current_cell);
            ctx->current_cell = TableCell{};
            ctx->in_table_cell = false;
        }
        break;
    case MD_BLOCK_TR:
        if (ctx->in_table_row)
        {
            ctx->table.table_rows.push_back(ctx->current_row);
            ctx->current_row = TableRow{};
            ctx->in_table_row = false;
        }
        break;
    case MD_BLOCK_TABLE:
        if (ctx->in_table)
        {
            ctx->blocks.push_back(ctx->table);
            ctx->table = MarkdownBlock{};
            ctx->in_table = false;
        }
        break;
    case MD_BLOCK_QUOTE:
        if (ctx->quote_depth > 0) --ctx->quote_depth;
        break;
    case MD_BLOCK_UL:
    case MD_BLOCK_OL:
        if (ctx->list_depth > 0) --ctx->list_depth;
        if (!ctx->ordered_stack.empty()) ctx->ordered_stack.pop_back();
        if (!ctx->ordered_counter.empty()) ctx->ordered_counter.pop_back();
        break;
    default:
        break;
    }
    return 0;
}

int enter_span(MD_SPANTYPE type, void*, void* userdata)
{
    auto* ctx = static_cast<ParseContext*>(userdata);
    switch (type)
    {
    case MD_SPAN_STRONG: ctx->style.strong = true; break;
    case MD_SPAN_EM: ctx->style.em = true; break;
    case MD_SPAN_CODE: ctx->style.code = true; break;
    case MD_SPAN_A: ctx->style.link = true; break;
    case MD_SPAN_IMG:
        ctx->style.image = true;
        append_text(*ctx, "[image: ", ctx->style);
        break;
    case MD_SPAN_DEL: ctx->style.strike = true; break;
    case MD_SPAN_U: ctx->style.underline = true; break;
    case MD_SPAN_LATEXMATH:
    case MD_SPAN_LATEXMATH_DISPLAY:
        ctx->style.math = true;
        append_text(*ctx, "$", ctx->style);
        break;
    case MD_SPAN_WIKILINK: ctx->style.link = true; break;
    default: break;
    }
    return 0;
}

int leave_span(MD_SPANTYPE type, void*, void* userdata)
{
    auto* ctx = static_cast<ParseContext*>(userdata);
    switch (type)
    {
    case MD_SPAN_STRONG: ctx->style.strong = false; break;
    case MD_SPAN_EM: ctx->style.em = false; break;
    case MD_SPAN_CODE: ctx->style.code = false; break;
    case MD_SPAN_A: ctx->style.link = false; break;
    case MD_SPAN_IMG:
        append_text(*ctx, "]", ctx->style);
        ctx->style.image = false;
        break;
    case MD_SPAN_DEL: ctx->style.strike = false; break;
    case MD_SPAN_U: ctx->style.underline = false; break;
    case MD_SPAN_LATEXMATH:
    case MD_SPAN_LATEXMATH_DISPLAY:
        append_text(*ctx, "$", ctx->style);
        ctx->style.math = false;
        break;
    case MD_SPAN_WIKILINK: ctx->style.link = false; break;
    default: break;
    }
    return 0;
}

int text_callback(MD_TEXTTYPE type, const MD_CHAR* text, MD_SIZE size, void* userdata)
{
    auto* ctx = static_cast<ParseContext*>(userdata);
    if (!ctx->collecting && !ctx->in_table_cell) return 0;
    if (type == MD_TEXT_BR || type == MD_TEXT_SOFTBR)
    {
        append_text(*ctx, "\n", ctx->style);
        return 0;
    }
    SpanStyle style = ctx->style;
    if (type == MD_TEXT_CODE || ctx->current.kind == BlockKind::CodeBlock)
    {
        style.code = true;
    }
    append_text(*ctx, std::string(text, text + size), style);
    return 0;
}

std::vector<MarkdownBlock> parse_markdown(const std::string& text)
{
    ParseContext ctx;
    MD_PARSER parser{};
    parser.abi_version = 0;
    parser.flags = MD_DIALECT_GITHUB | MD_FLAG_NOHTML |
                   MD_FLAG_LATEXMATHSPANS | MD_FLAG_WIKILINKS |
                   MD_FLAG_UNDERLINE | MD_FLAG_HARD_SOFT_BREAKS;
    parser.enter_block = enter_block;
    parser.leave_block = leave_block;
    parser.enter_span = enter_span;
    parser.leave_span = leave_span;
    parser.text = text_callback;

    int rc = md_parse(text.c_str(), static_cast<MD_SIZE>(text.size()), &parser, &ctx);
    finish_current(ctx);
    if (rc != 0 || ctx.blocks.empty())
    {
        MarkdownBlock fallback;
        fallback.kind = BlockKind::Paragraph;
        fallback.spans.push_back({text, {}});
        ctx.blocks.push_back(fallback);
    }
    return ctx.blocks;
}

Fl_Color bubble_color(ChatBubbleView::Role role)
{
    if (role == ChatBubbleView::Role::User) return fl_rgb_color(149, 236, 105);
    if (role == ChatBubbleView::Role::System) return fl_rgb_color(255, 244, 204);
    return fl_rgb_color(255, 255, 255);
}

Fl_Color avatar_color(ChatBubbleView::Role role)
{
    if (role == ChatBubbleView::Role::User) return fl_rgb_color(70, 160, 70);
    if (role == ChatBubbleView::Role::System) return fl_rgb_color(210, 160, 40);
    return fl_rgb_color(80, 135, 220);
}

const char* avatar_text(ChatBubbleView::Role role)
{
    if (role == ChatBubbleView::Role::User) return "U";
    if (role == ChatBubbleView::Role::System) return "!";
    return "AI";
}

std::size_t table_column_count(const MarkdownBlock& table)
{
    std::size_t cols = 0;
    for (const auto& row : table.table_rows)
    {
        cols = std::max(cols, row.cells.size());
    }
    return cols;
}

MarkdownBlock cell_as_block(const TableCell& cell)
{
    MarkdownBlock block;
    block.kind = BlockKind::Paragraph;
    block.spans = cell.spans;
    if (cell.header)
    {
        for (auto& span : block.spans)
        {
            span.style.strong = true;
        }
    }
    return block;
}

int measure_table_height(const MarkdownBlock& table, int text_width)
{
    std::size_t cols = table_column_count(table);
    if (cols == 0) return kLineHeight;
    int cell_w = std::max(40, text_width / static_cast<int>(cols));
    int height = 0;
    for (const auto& row : table.table_rows)
    {
        int row_h = kLineHeight + 10;
        for (const auto& cell : row.cells)
        {
            auto lines = layout_spans(cell_as_block(cell), cell_w - 10);
            row_h = std::max(row_h, static_cast<int>(lines.size()) * kLineHeight + 10);
        }
        height += row_h;
    }
    return height + 4;
}

int measure_blocks_width(const std::vector<MarkdownBlock>& blocks, int max_text_width)
{
    int width = 50;
    for (const auto& block : blocks)
    {
        if (block.kind == BlockKind::Rule)
        {
            width = std::max(width, 100);
            continue;
        }
        if (block.kind == BlockKind::Table)
        {
            width = std::max(width, max_text_width);
            continue;
        }
        int indent = block.kind == BlockKind::ListItem ? 18 : 0;
        if (block.kind == BlockKind::Quote) indent += 10;
        auto lines = layout_spans(block, std::max(40, max_text_width - indent));
        for (const auto& line : lines)
        {
            width = std::max(width, line.width + indent);
        }
    }
    return std::min(max_text_width, width);
}

int measure_blocks_height(const std::vector<MarkdownBlock>& blocks, int text_width)
{
    int h = 0;
    for (const auto& block : blocks)
    {
        if (block.kind == BlockKind::Rule)
        {
            h += 12;
            continue;
        }
        if (block.kind == BlockKind::Table)
        {
            h += measure_table_height(block, text_width) + 6;
            continue;
        }
        int indent = block.kind == BlockKind::ListItem ? 18 : 0;
        if (block.kind == BlockKind::Quote) indent += 10;
        auto lines = layout_spans(block, std::max(40, text_width - indent));
        h += static_cast<int>(lines.size()) * block_line_height(block) + 6;
    }
    return std::max(kLineHeight, h);
}

} // namespace

class ChatBubbleView::Canvas : public Fl_Widget
{
public:
    Canvas(int x, int y, int w, int h, ChatBubbleView& owner)
        : Fl_Widget(x, y, w, h), owner_(owner)
    {
    }

    void draw() override
    {
        fl_push_clip(x(), y(), w(), h());
        fl_color(fl_rgb_color(245, 245, 245));
        fl_rectf(x(), y(), w(), h());

        int y_cursor = y() + kPadding;
        const int content_w = w() - kPadding * 2;
        const int max_bubble_w = std::max(120, content_w * 2 / 3);

        for (const auto& msg : owner_.messages_)
        {
            auto blocks = parse_markdown(msg.text.empty() ? " " : msg.text);
            int max_body_w = max_bubble_w - kBubblePadX * 2;
            int body_w = std::max(50, measure_blocks_width(blocks, max_body_w));
            int body_h = measure_blocks_height(blocks, body_w);
            int bubble_w = body_w + kBubblePadX * 2;
            int bubble_h = body_h + kBubblePadY * 2;
            int row_h = std::max(kAvatar, bubble_h);

            bool right = msg.role == ChatBubbleView::Role::User;
            int avatar_x = right ? x() + w() - kPadding - kAvatar : x() + kPadding;
            int bubble_x = right ? avatar_x - kGap - bubble_w : avatar_x + kAvatar + kGap;
            int bubble_y = y_cursor;
            int avatar_y = y_cursor;

            fl_color(avatar_color(msg.role));
            fl_pie(avatar_x, avatar_y, kAvatar, kAvatar, 0, 360);
            fl_color(FL_WHITE);
            fl_font(FL_HELVETICA_BOLD, 12);
            fl_draw(avatar_text(msg.role), avatar_x, avatar_y, kAvatar, kAvatar, FL_ALIGN_CENTER);

            fl_color(bubble_color(msg.role));
            fl_rectf(bubble_x, bubble_y, bubble_w, bubble_h);
            fl_color(fl_rgb_color(220, 220, 220));
            fl_rect(bubble_x, bubble_y, bubble_w, bubble_h);

            int tx = bubble_x + kBubblePadX;
            int ty = bubble_y + kBubblePadY;
            for (const auto& block : blocks)
            {
                if (block.kind == BlockKind::Rule)
                {
                    fl_color(fl_rgb_color(200, 200, 200));
                    fl_line(tx, ty + 6, tx + body_w, ty + 6);
                    ty += 12;
                    continue;
                }
                if (block.kind == BlockKind::Table)
                {
                    std::size_t cols = table_column_count(block);
                    if (cols == 0)
                    {
                        continue;
                    }
                    int cell_w = std::max(40, body_w / static_cast<int>(cols));
                    int table_x = tx;
                    int table_y = ty;
                    for (const auto& row : block.table_rows)
                    {
                        int row_h = kLineHeight + 10;
                        std::vector<std::vector<RenderLine>> laid_out;
                        for (const auto& cell : row.cells)
                        {
                            auto lines = layout_spans(cell_as_block(cell), cell_w - 10);
                            row_h = std::max(row_h, static_cast<int>(lines.size()) * kLineHeight + 10);
                            laid_out.push_back(std::move(lines));
                        }
                        for (std::size_t c = 0; c < cols; ++c)
                        {
                            int cx = table_x + static_cast<int>(c) * cell_w;
                            bool header = c < row.cells.size() && row.cells[c].header;
                            fl_color(header ? fl_rgb_color(235, 240, 248) : fl_rgb_color(250, 250, 250));
                            fl_rectf(cx, ty, cell_w, row_h);
                            fl_color(fl_rgb_color(205, 205, 205));
                            fl_rect(cx, ty, cell_w, row_h);
                            if (c < laid_out.size())
                            {
                                int cy = ty + 5;
                                for (const auto& line : laid_out[c])
                                {
                                    int x_cursor = cx + 5;
                                    MD_ALIGN align = row.cells[c].align;
                                    if (align == MD_ALIGN_CENTER)
                                    {
                                        x_cursor = cx + std::max(5, (cell_w - line.width) / 2);
                                    }
                                    else if (align == MD_ALIGN_RIGHT)
                                    {
                                        x_cursor = cx + std::max(5, cell_w - line.width - 5);
                                    }
                                    for (const auto& span : line.spans)
                                    {
                                        int font = span_font(span.style, cell_as_block(row.cells[c]));
                                        fl_font(font, kTextSize);
                                        if (span.style.code || span.style.math)
                                        {
                                            fl_color(span.style.math ? fl_rgb_color(246, 240, 255) : fl_rgb_color(235, 235, 235));
                                            fl_rectf(x_cursor - 2, cy + 2, span.width + 4, kLineHeight - 3);
                                        }
                                        Fl_Color text_color = FL_BLACK;
                                        if (span.style.link) text_color = fl_rgb_color(40, 100, 200);
                                        if (span.style.image) text_color = fl_rgb_color(120, 90, 20);
                                        if (span.style.math) text_color = fl_rgb_color(90, 50, 150);
                                        fl_color(text_color);
                                        fl_draw(span.text.c_str(), x_cursor, cy + kTextSize);
                                        if (span.style.link || span.style.underline)
                                        {
                                            fl_line(x_cursor, cy + kTextSize + 2, x_cursor + span.width, cy + kTextSize + 2);
                                        }
                                        if (span.style.strike)
                                        {
                                            fl_line(x_cursor, cy + kLineHeight / 2, x_cursor + span.width, cy + kLineHeight / 2);
                                        }
                                        x_cursor += span.width;
                                    }
                                    cy += kLineHeight;
                                }
                            }
                        }
                        ty += row_h;
                    }
                    (void)table_y;
                    ty += 6;
                    continue;
                }

                int line_h = block_line_height(block);
                int indent = block.kind == BlockKind::ListItem ? 18 : 0;
                if (block.kind == BlockKind::Quote) indent += 10;
                auto lines = layout_spans(block, std::max(40, body_w - indent));

                if (block.kind == BlockKind::CodeBlock)
                {
                    int code_h = static_cast<int>(lines.size()) * line_h + 8;
                    fl_color(fl_rgb_color(242, 242, 242));
                    fl_rectf(tx, ty, body_w, code_h);
                    fl_color(fl_rgb_color(210, 210, 210));
                    fl_rect(tx, ty, body_w, code_h);
                    ty += 4;
                }
                else if (block.kind == BlockKind::Quote)
                {
                    fl_color(fl_rgb_color(180, 180, 180));
                    fl_rectf(tx, ty, 4, static_cast<int>(lines.size()) * line_h);
                }

                if (block.kind == BlockKind::ListItem)
                {
                    fl_color(FL_BLACK);
                    fl_font(FL_HELVETICA, kTextSize);
                    std::string marker;
                    if (block.task)
                    {
                        marker = block.task_checked ? "☑" : "☐";
                    }
                    else if (block.ordered)
                    {
                        marker = std::to_string(block.list_number) + ".";
                    }
                    else
                    {
                        marker = "•";
                    }
                    fl_draw(marker.c_str(), tx, ty + kTextSize);
                }

                for (const auto& line : lines)
                {
                    int x_cursor = tx + indent;
                    for (const auto& span : line.spans)
                    {
                        int font = span_font(span.style, block);
                        int size = block_size(block);
                        fl_font(font, size);
                        if ((span.style.code || span.style.math) && block.kind != BlockKind::CodeBlock)
                        {
                            fl_color(span.style.math ? fl_rgb_color(246, 240, 255) : fl_rgb_color(235, 235, 235));
                            fl_rectf(x_cursor - 2, ty + 2, span.width + 4, line_h - 3);
                        }
                        Fl_Color text_color = FL_BLACK;
                        if (span.style.link) text_color = fl_rgb_color(40, 100, 200);
                        if (span.style.image) text_color = fl_rgb_color(120, 90, 20);
                        if (span.style.math) text_color = fl_rgb_color(90, 50, 150);
                        fl_color(text_color);
                        fl_draw(span.text.c_str(), x_cursor, ty + size);
                        if (span.style.link || span.style.underline)
                        {
                            fl_line(x_cursor, ty + size + 2, x_cursor + span.width, ty + size + 2);
                        }
                        if (span.style.strike)
                        {
                            fl_line(x_cursor, ty + line_h / 2, x_cursor + span.width, ty + line_h / 2);
                        }
                        x_cursor += span.width;
                    }
                    ty += line_h;
                }
                ty += 6;
            }

            y_cursor += row_h + kGap;
        }

        fl_pop_clip();
    }

private:
    ChatBubbleView& owner_;
};

ChatBubbleView::ChatBubbleView(int x, int y, int w, int h, const char* label)
    : Fl_Scroll(x, y, w, h, label)
{
    type(Fl_Scroll::VERTICAL_ALWAYS);
    color(fl_rgb_color(245, 245, 245));
    box(FL_FLAT_BOX);

    canvas_ = new Canvas(x, y, w - scrollbar.w(), h, *this);
    end();
}

int ChatBubbleView::add_message(Role role, const std::string& text)
{
    messages_.push_back({role, text});
    relayout();
    scroll_to_bottom();
    return static_cast<int>(messages_.size()) - 1;
}

void ChatBubbleView::append_to_message(int index, const std::string& text, bool defer_layout)
{
    if (index < 0 || static_cast<std::size_t>(index) >= messages_.size())
    {
        return;
    }
    messages_[static_cast<std::size_t>(index)].text += text;
    if (defer_layout)
    {
        layout_pending_ = true;
        return;
    }
    relayout();
    scroll_to_bottom();
}

void ChatBubbleView::flush_layout()
{
    if (!layout_pending_)
    {
        return;
    }
    layout_pending_ = false;
    relayout();
    scroll_to_bottom();
}

void ChatBubbleView::clear_messages()
{
    messages_.clear();
    relayout();
    scroll_to_bottom();
}

void ChatBubbleView::relayout()
{
    int content_w = std::max(100, w() - scrollbar.w() - kPadding * 2);
    int max_bubble_w = std::max(120, content_w * 2 / 3);
    int max_body_w = max_bubble_w - kBubblePadX * 2;
    int total_h = kPadding;
    for (const auto& msg : messages_)
    {
        auto blocks = parse_markdown(msg.text.empty() ? " " : msg.text);
        int body_w = std::max(50, measure_blocks_width(blocks, max_body_w));
        int bubble_h = measure_blocks_height(blocks, body_w) + kBubblePadY * 2;
        total_h += std::max(kAvatar, bubble_h) + kGap;
    }
    total_h = std::max(total_h + kPadding, h());

    // Preserve the visual scroll offset while the canvas height changes.
    // Fl_Scroll stores the logical yposition separately from the child widget
    // coordinates; resizing the child back to y() would make the scrollbar state
    // and canvas position disagree, which causes flicker on the next scroll.
    canvas_->resize(x(), y() - yposition(), w() - scrollbar.w(), total_h);
    init_sizes();
    redraw();
}

void ChatBubbleView::scroll_to_bottom()
{
    int viewport_h = h() - scrollbar.w();
    int bottom = std::max(0, canvas_->h() - viewport_h);

    scroll_to(0, bottom);

    if (bottom_scroll_pending_)
    {
        return;
    }
    bottom_scroll_pending_ = true;

    // Scrollbar geometry can be finalized one event-loop tick later on some
    // platforms. A second delayed scroll keeps streaming output pinned to the
    // bottom instead of drifting a few lines above it.
    Fl::add_timeout(0.01, [](void* data) {
        auto* view = static_cast<ChatBubbleView*>(data);
        view->bottom_scroll_pending_ = false;
        int h = view->h() - view->scrollbar.w();
        int b = std::max(0, view->canvas_->h() - h);
        view->scroll_to(0, b);
        view->redraw();
    }, this);
}

} // namespace smart_app::ui
